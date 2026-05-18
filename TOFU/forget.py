import os
import re
import json
import shutil

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

@hydra.main(version_base=None, config_path="config", config_name="forget")
def main(cfg):

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

    batch_size                  = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    steps_per_epoch = len(torch_format_dataset) // (
        batch_size * gradient_accumulation_steps * num_devices
    )
    max_steps = int(cfg.num_epochs * len(torch_format_dataset)) // (
        batch_size * gradient_accumulation_steps * num_devices
    )
    print(f"max_steps: {max_steps}")

    # ── Training args ─────────────────────────────────────────────────────────
    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=max(1, steps_per_epoch),
        max_steps=max_steps,
        learning_rate=cfg.lr,
        bf16=True,
        bf16_full_eval=True,
        logging_steps=max(1, max_steps // 20),
        logging_dir=f"{cfg.save_dir}/logs",
        output_dir=cfg.save_dir,
        optim="paged_adamw_32bit",
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

    oracle_model = None

    if path_found:
        print("Loading from checkpoint")
        config = AutoConfig.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            config=config,
            attn_implementation="flash_attention_2" if model_cfg["flash_attention2"] == "true" else "eager",
            torch_dtype=torch.bfloat16,
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
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
    else:
        print("Loading after merge and unload")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation="flash_attention_2" if model_cfg["flash_attention2"] == "true" else "eager",
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
        model = PeftModel.from_pretrained(model, model_id=cfg.model_path)
        model = model.merge_and_unload()
        model.save_pretrained(cfg.model_path)

    # Hot fix: https://discuss.huggingface.co/t/help-with-llama-2-finetuning-setup/50035
    model.generation_config.do_sample = True

    # ── LoRA ──────────────────────────────────────────────────────────────────
    lora_targets = get_lora_targets(cfg.model_family, cfg.LoRA.targets)

    lora_config = LoraConfig(
        r=cfg.LoRA.r,
        lora_alpha=cfg.LoRA.alpha,
        target_modules=lora_targets,
        lora_dropout=cfg.LoRA.dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if cfg.LoRA.r != 0:
        model = get_peft_model(model, lora_config)
        print_trainable_parameters(model)

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = CustomTrainerForgetting(
        model=model,
        processing_class=tokenizer,
        train_dataset=torch_format_dataset,
        eval_dataset=torch_format_dataset,
        compute_metrics=None,
        args=training_args,
        data_collator=custom_data_collator_forget,
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
    )

    model.config.use_cache = False

    # ── Train / Eval ──────────────────────────────────────────────────────────
    if cfg.eval_only:
        trainer.evaluate()
    else:
        trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    if cfg.save_model and not cfg.eval_only:
        if cfg.importance_file:
            model = model.unload()
        model.save_pretrained(cfg.save_dir)
        tokenizer.save_pretrained(cfg.save_dir)

    # ── Cleanup deepspeed checkpoints ─────────────────────────────────────────
    if local_rank == 0:
        for ckpt_dir in Path(cfg.save_dir).glob("checkpoint-*"):
            for global_step_dir in ckpt_dir.glob("global_step*"):
                shutil.rmtree(global_step_dir)


if __name__ == "__main__":
    main()