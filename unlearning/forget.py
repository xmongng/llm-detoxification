import os
import re
import json
import shutil
import math
import copy
import traceback

import torch
import hydra
import deepspeed
import transformers
import random
import numpy as np

from pathlib import Path
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed
from peft import LoraConfig, get_peft_model, PeftModel

from data_module import (
    TextForgetDatasetQA,
    TextForgetDatasetDPOQA,
    LocalTextForgetDatasetQA,
    LocalTextForgetDatasetIDK,
)
from dataloader import CustomTrainerForgetting, custom_data_collator_forget
from utils import get_model_identifiers_from_yaml
from codeeraser import CodeEraserDataset, codeeraser_collator
from prod import PRODDataset, prod_collator
from experiment_tracking import (RunRecorder, ExperimentCallback, unique_run_dir,
                                 atomic_json, environment_manifest, file_sha256)
from lora_utils import resolve_lora, configure_trainable_model, parameter_summary


# ── Helpers ────────────────────────────────────────────────────────────────────

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )


def build_dataset(cfg, tokenizer, max_length):
    """
    Local JSONL  → LocalTextForgetDatasetQA / LocalTextForgetDatasetIDK
    HuggingFace Hub → TextForgetDatasetQA / TextForgetDatasetDPOQA  (backward-compatible)
    """
    use_local = bool(
        getattr(cfg, "forget_data_path", None) and
        getattr(cfg, "retain_data_path", None)
    )

    if cfg.forget_loss == 'prod':
        if not cfg.forget_data_path:
            raise ValueError('PROD requires forget_data_path JSONL.')
        return PRODDataset(cfg.forget_data_path, cfg.retain_data_path, tokenizer,
                           get_model_identifiers_from_yaml(cfg.model_family), max_length)

    if cfg.forget_loss == 'codeeraser':
        if not use_local:
            raise ValueError('CodeEraser requires local forget_data_path and retain_data_path JSONL files.')
        return CodeEraserDataset(cfg.forget_data_path, cfg.retain_data_path, tokenizer,
                                get_model_identifiers_from_yaml(cfg.model_family), max_length,
                                getattr(cfg, 'retain_max_length', 128))

    # ── Local JSONL ───────────────────────────────────────────────────────────
    if use_local:
        print(f"[Dataset] Local JSONL — forget: {cfg.forget_data_path} | retain: {cfg.retain_data_path}")

        if cfg.forget_loss == "idk":
            return LocalTextForgetDatasetIDK(
                forget_data_path=cfg.forget_data_path,
                retain_data_path=cfg.retain_data_path,
                tokenizer=tokenizer,
                model_family=cfg.model_family,
                max_length=max_length,
                idk_file=cfg.idk_file,
            )
        else:
            # grad_diff | grad_ascent | IHL | KL | npo_grad_diff | simnpo_gdr | simnpo
            return LocalTextForgetDatasetQA(
                forget_data_path=cfg.forget_data_path,
                retain_data_path=cfg.retain_data_path,
                tokenizer=tokenizer,
                model_family=cfg.model_family,
                max_length=max_length,
                loss_type=cfg.forget_loss,
            )

    # ── HuggingFace Hub (backward-compatible) ─────────────────────────────────
    print(f"[Dataset] Using HuggingFace Hub — data_path: {cfg.data_path} | split: {cfg.split}")

    if cfg.forget_loss == "dpo":
        return TextForgetDatasetDPOQA(
            cfg.data_path,
            tokenizer=tokenizer,
            model_family=cfg.model_family,
            max_length=max_length,
            split=cfg.split,
        )
    else:
        return TextForgetDatasetQA(
            cfg.data_path,
            tokenizer=tokenizer,
            model_family=cfg.model_family,
            max_length=max_length,
            split=cfg.split,
            loss_type=cfg.forget_loss,
        )


def get_lora_targets(model_family, targets):
    # ── Shared architecture target sets ───────────────────────────────────────
    # Qwen2 / Qwen2.5 / Qwen3 architecture (same projection layer names)
    _qwen2_targets = {
        "all":       ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
        "self_attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp":       ["gate_proj", "up_proj", "down_proj"],
    }

    # DeepSeek-Coder uses LLaMA-style architecture (same projection layer names)
    _deepseek_targets = {
        "all":       ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
        "self_attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp":       ["gate_proj", "up_proj", "down_proj"],
    }

    target_map = {
        # ── Qwen2.5-Coder family ──────────────────────────────────────────────
        "qwen2.5-coder-0.5b": _qwen2_targets,
        "qwen2.5-coder-1.5b": _qwen2_targets,
        "qwen2.5-coder-3b":   _qwen2_targets,
        "qwen2.5-coder-7b":   _qwen2_targets,
        # ── Qwen3 family ─────────────────────────────────────────────────────
        "qwen3-4b":           _qwen2_targets,
        # ── DeepSeek-Coder family ─────────────────────────────────────────────
        "deepseek-coder-1.3b": _deepseek_targets,
        "deepseek-coder-6.7b": _deepseek_targets,
    }

    if model_family not in target_map:
        raise NotImplementedError(
            f"model_family '{model_family}' is not supported. "
            f"Available: {list(target_map.keys())}"
        )
    if targets not in target_map[model_family]:
        raise NotImplementedError(
            f"Invalid LoRA targets '{targets}' for model family '{model_family}'. "
            f"Available: {list(target_map[model_family].keys())}"
        )

    return target_map[model_family][targets]


# ── Main ───────────────────────────────────────────────────────────────────────

def run_training(cfg, recorder=None):

    use_lora = resolve_lora(cfg.forget_loss, cfg.LoRA.enabled, cfg.LoRA.r)

    if cfg.forget_loss in ('codeeraser', 'prod'):
        if cfg.importance_file:
            raise ValueError(f'{cfg.forget_loss} requires importance_file=null.')
        if cfg.eval_while_train or cfg.eval_only:
            raise ValueError('Use evals/eval_memorize.py; the legacy TOFU evaluator is not supported for this method.')

    # ── FILA mode ─────────────────────────────────────────────────────────────
    if cfg.importance_file:
        print("Using FILA...")
        cfg.save_dir = cfg.save_dir.replace(cfg.forget_loss, cfg.forget_loss + "_FILA")

    print("######################")
    print("Saving to:", cfg.save_dir)
    print("######################")

    # ── Environment ───────────────────────────────────────────────────────────
    num_devices = int(os.environ.get("WORLD_SIZE", 1))
    print(f"num_devices: {num_devices}")

    local_rank = 0
    device_map = {"": 0}
    if os.environ.get("LOCAL_RANK") is not None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_map = {"": local_rank}

    set_seed(cfg.seed)
    os.environ["WANDB_DISABLED"] = "true"

    # ── Model config ──────────────────────────────────────────────────────────
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id  = model_cfg["hf_key"]
    if cfg.model_path is None:
        cfg.model_path = model_cfg["ft_model_path"]

    # ── Save dir ──────────────────────────────────────────────────────────────
    if os.path.exists(cfg.save_dir) and not cfg.overwrite_dir:
        print("Directory already exists, exiting.")
        exit()

    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    if local_rank == 0:
        with open(f"{cfg.save_dir}/config.yaml", "w") as f:
            OmegaConf.save(cfg, f)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Dataset ───────────────────────────────────────────────────────────────
    max_length = getattr(cfg, 'max_length', 500)
    torch_format_dataset = build_dataset(cfg, tokenizer, max_length)
    tracking_callback = None
    if recorder is not None:
        tracking_callback = ExperimentCallback(
            recorder, torch_format_dataset, tokenizer, model_cfg,
            OmegaConf.to_container(cfg.tracking, resolve=True), num_devices)

    batch_size                  = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    steps_per_epoch = len(torch_format_dataset) // (
        batch_size * gradient_accumulation_steps * num_devices
    )
    max_steps = int(cfg.num_epochs * len(torch_format_dataset)) // (
        batch_size * gradient_accumulation_steps * num_devices
    )
    if cfg.forget_loss == 'codeeraser':
        # One forget update followed by one retain update per effective batch.
        steps_per_epoch = 2 * math.ceil(len(torch_format_dataset) / (
            batch_size * gradient_accumulation_steps * num_devices))
        max_steps = int(cfg.num_epochs) * steps_per_epoch
        if max_steps <= 0 or int(cfg.num_epochs) != cfg.num_epochs:
            raise ValueError('CodeEraser num_epochs must be a positive integer.')
    if cfg.forget_loss == 'prod':
        steps_per_epoch = math.ceil(math.ceil(len(torch_format_dataset) / (batch_size * num_devices))
                                    / gradient_accumulation_steps)
        max_steps = int(cfg.num_epochs) * steps_per_epoch
        if max_steps <= 0 or int(cfg.num_epochs) != cfg.num_epochs:
            raise ValueError('PROD num_epochs must be a positive integer.')
    print(f"max_steps: {max_steps}")

    precision = getattr(cfg, 'precision', 'auto')
    if precision == 'auto':
        precision = 'bf16' if torch.cuda.is_bf16_supported() else 'fp16'
    if precision not in ('bf16', 'fp16', 'fp32'):
        raise ValueError('precision must be auto, bf16, fp16, or fp32.')
    model_dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[precision]

    # ── Training args ─────────────────────────────────────────────────────────
    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=0 if cfg.forget_loss in ('codeeraser', 'prod') else max(1, steps_per_epoch),
        max_steps=max_steps,
        learning_rate=cfg.lr,
        lr_scheduler_type='constant' if cfg.forget_loss in ('codeeraser', 'prod') else 'linear',
        bf16=precision == 'bf16',
        fp16=precision == 'fp16',
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        logging_steps=cfg.tracking.log_every_steps if recorder is not None else max(1, max_steps // 20),
        logging_dir=f"{cfg.save_dir}/logs",
        output_dir=cfg.save_dir,
        optim='adamw_torch' if cfg.forget_loss in ('codeeraser', 'prod') else 'paged_adamw_32bit',
        adam_beta2=0.98 if cfg.forget_loss == 'codeeraser' else 0.999,
        max_grad_norm=0.0 if cfg.forget_loss in ('codeeraser', 'prod') else 1.0,
        remove_unused_columns=cfg.forget_loss not in ('codeeraser', 'prod'),
        save_strategy="steps" if cfg.save_model and not cfg.eval_only else "no",
        save_steps=steps_per_epoch,
        save_only_model=True,
        ddp_find_unused_parameters=False,
        deepspeed="config/ds_config.json",
        weight_decay=cfg.weight_decay,
        eval_steps=steps_per_epoch,
        eval_strategy="steps" if cfg.eval_while_train else "no",
        seed=cfg.seed,
    )

    # ── Load model ────────────────────────────────────────────────────────────
    path_found = False
    for fname in os.listdir(cfg.model_path):
        if re.search(r"pytorch.*\.bin", fname) or re.search(r"model-*\.safetensors", fname):
            path_found = True
            break
    checkpoint = Path(cfg.model_path)
    path_found = any(checkpoint.glob('model*.safetensors')) or any(checkpoint.glob('pytorch_model*.bin'))

    oracle_model = None

    if cfg.forget_loss in ('codeeraser', 'prod'):
        # Load the actual starting checkpoint, including adapter-only checkpoints.
        # Freeze an exact copy before any update, as required by SSCU's KL term.
        config = AutoConfig.from_pretrained(cfg.model_path if path_found else model_id)
        if cfg.forget_loss == 'codeeraser' and hasattr(config, 'attention_dropout'):
            config.attention_dropout = 0.0
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path if path_found else model_id,
            config=config, torch_dtype=model_dtype,
            attn_implementation='eager', trust_remote_code=True,
        )
        if not path_found:
            model = PeftModel.from_pretrained(model, cfg.model_path).merge_and_unload()
        oracle_model = copy.deepcopy(model)
        oracle_model.requires_grad_(False)
        oracle_model.eval()
    elif path_found:
        print("Loading from checkpoint")
        config = AutoConfig.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            config=config,
            attn_implementation="flash_attention_2" if model_cfg["flash_attention2"] == "true" else "eager",
            torch_dtype=model_dtype,
            trust_remote_code=True,
        )
        # Oracle model cần thiết cho KL và tất cả biến thể NPO
        if cfg.forget_loss == "KL" or (
            "npo" in cfg.forget_loss and "simnpo" not in cfg.forget_loss
        ):
            oracle_model = AutoModelForCausalLM.from_pretrained(
                cfg.model_path,
                config=config,
                attn_implementation="flash_attention_2" if model_cfg["flash_attention2"] == "true" else "eager",
                torch_dtype=model_dtype,
                trust_remote_code=True,
            )
    else:
        print("Loading after merge and unload")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation="flash_attention_2" if model_cfg["flash_attention2"] == "true" else "eager",
            torch_dtype=model_dtype,
            device_map=device_map,
        )
        model = PeftModel.from_pretrained(model, model_id=cfg.model_path)
        model = model.merge_and_unload()

    if oracle_model is None and (cfg.forget_loss == 'KL' or (
            'npo' in cfg.forget_loss and 'simnpo' not in cfg.forget_loss)):
        oracle_model = copy.deepcopy(model)
        oracle_model.requires_grad_(False)
        oracle_model.eval()

    # Hot fix: https://discuss.huggingface.co/t/help-with-llama-2-finetuning-setup/50035
    model.generation_config.do_sample = True

    # ── LoRA ──────────────────────────────────────────────────────────────────
    if use_lora:
        if not path_found:
            # A new adapter needs the merged starting weights as its actual base.
            base_path = str(Path(cfg.save_dir).resolve() / 'base_model')
            if int(os.environ.get('RANK', 0)) == 0:
                model.save_pretrained(base_path)
                tokenizer.save_pretrained(base_path)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            model.name_or_path = base_path
            model.config._name_or_path = base_path
        lora_targets = get_lora_targets(cfg.model_family, cfg.LoRA.targets)
        lora_config = LoraConfig(
            r=cfg.LoRA.r,
            lora_alpha=cfg.LoRA.alpha,
            target_modules=lora_targets,
            lora_dropout=cfg.LoRA.dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = configure_trainable_model(model, True, lora_config, get_peft_model)
    else:
        model = configure_trainable_model(model, False)
    mode_summary = parameter_summary(model, use_lora)
    print(f'Parameter update mode: {mode_summary}')
    if int(os.environ.get('RANK', 0)) == 0:
        atomic_json(Path(cfg.save_dir) / 'parameter_training.json', mode_summary)
        if recorder is not None:
            atomic_json(recorder.directory / 'parameter_training.json', mode_summary)

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = CustomTrainerForgetting(
        model=model,
        processing_class=tokenizer,
        train_dataset=torch_format_dataset,
        eval_dataset=torch_format_dataset,
        compute_metrics=None,
        args=training_args,
        data_collator={'codeeraser': codeeraser_collator, 'prod': prod_collator}.get(
            cfg.forget_loss, custom_data_collator_forget),
        oracle_model=oracle_model,
        forget_loss=cfg.forget_loss,
        eval_cfg=cfg.eval,
        seed=cfg.seed,
        ref_policy=cfg.ref_policy,
        beta=cfg.beta,
        npo_coeff=cfg.npo_coeff,
        grad_diff_coeff=cfg.grad_diff_coeff,
        KL_coeff=cfg.KL_coeff,
        gamma=cfg.gamma,
        control_alpha=getattr(cfg, 'control_alpha', 1.0),
        control_lambda=getattr(cfg, 'control_lambda', 0.1),
        select_gamma=getattr(cfg, 'select_gamma', 0.1),
        prod_settings=OmegaConf.to_container(cfg.prod, resolve=True),
        callbacks=[tracking_callback] if tracking_callback is not None else None,
    )

    model.config.use_cache = False
    if recorder is not None and int(os.environ.get('RANK', 0)) == 0:
        atomic_json(recorder.directory / 'training_arguments.json', training_args.to_dict())
        OmegaConf.save(cfg, recorder.directory / 'config.yaml', resolve=True)
        atomic_json(recorder.directory / 'dataset_sizes.json', {
            'forget': len(torch_format_dataset),
            'retain': len(getattr(torch_format_dataset, 'retain',
                                  getattr(torch_format_dataset, 'retain_data', [])))})

    # ── Train / Eval ──────────────────────────────────────────────────────────
    if cfg.eval_only:
        trainer.evaluate()
    else:
        trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    if cfg.save_model and not cfg.eval_only:
        # Save the trained adapter in LoRA mode; unloading would discard updates.
        trainer.save_model(cfg.save_dir)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(cfg.save_dir)

    # ── Cleanup deepspeed checkpoints ─────────────────────────────────────────
    if int(os.environ.get('RANK', 0)) == 0:
        for ckpt_dir in Path(cfg.save_dir).glob("checkpoint-*"):
            for global_step_dir in ckpt_dir.glob("global_step*"):
                shutil.rmtree(global_step_dir)


@hydra.main(version_base=None, config_path="config", config_name="forget")
def main(cfg):
    use_lora = resolve_lora(cfg.forget_loss, cfg.LoRA.enabled, cfg.LoRA.r)
    if not cfg.tracking.enabled:
        return run_training(cfg)
    from accelerate import PartialState
    distributed = PartialState()
    rank = distributed.process_index
    local_rank = distributed.local_process_index
    requested_save_dir = str(cfg.save_dir)
    mode = 'lora' if use_lora else 'full'
    paths = [str(unique_run_dir(cfg.tracking.output_dir, f'{cfg.model_family}_{cfg.forget_loss}_{mode}'))
             if rank == 0 else None]
    if torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(paths, src=0)
    run_dir = Path(paths[0])
    # Shared run ID, independent rank files, and checkpoints isolated from older runs.
    cfg.save_dir = str(run_dir / 'model')
    if rank == 0:
        manifest = environment_manifest()
        manifest.update(run_id=run_dir.name, requested_save_dir=requested_save_dir,
                        resolved_save_dir=cfg.save_dir, world_size=distributed.num_processes,
                        method=cfg.forget_loss, model_family=cfg.model_family, datasets={},
                        parameter_update_mode=mode, lora_rank=cfg.LoRA.r if use_lora else 0)
        for split in ('forget', 'retain'):
            path = getattr(cfg, f'{split}_data_path', None)
            if path and Path(path).is_file():
                manifest['datasets'][split] = {'path': str(Path(path).resolve()), 'sha256': file_sha256(path)}
                if cfg.tracking.copy_datasets:
                    (run_dir / 'inputs').mkdir(exist_ok=True)
                    shutil.copy2(path, run_dir / 'inputs' / f'{split}.jsonl')
        atomic_json(run_dir / 'manifest.json', manifest)
        OmegaConf.save(cfg, run_dir / 'config.yaml', resolve=True)
        source_root = Path(__file__).parent
        for pattern in ('*.py', 'config/*', 'evals/*.py'):
            for source in source_root.glob(pattern):
                if source.is_file():
                    destination = run_dir / 'source' / source.relative_to(source_root)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
        print(f'Experiment artifacts: {run_dir}', flush=True)
    distributed.wait_for_everyone()
    recorder = RunRecorder(run_dir, rank, local_rank, cfg.tracking.resource_interval_seconds)
    recorder.start()
    status, error = 'completed', None
    try:
        run_training(cfg, recorder)
    except BaseException:
        status, error = 'failed', traceback.format_exc()
        raise
    finally:
        recorder.finish(status, error)
    distributed.wait_for_everyone()
    if rank == 0:
        from summarize_run import summarize_run
        summarize_run(run_dir)


if __name__ == "__main__":
    main()
