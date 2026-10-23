import torch
from torch import nn
from torch import Tensor, tensor
from transformers import Trainer
import torch.nn.functional as F
import copy, os
import deepspeed
from evaluate_util import get_dataloader, get_all_evals
import copy
import json 
from pathlib import Path
from data_module import get_batch_loss 
from utils import merge_dicts, interleave_eval_result_dict, get_forget_quality, get_model_utility
import numpy as np
from scipy.stats import ks_2samp, hmean
import csv 
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from codeeraser import AlternatingBatchSampler, forget_objective, retain_objective
from prod import prod_objective

def printll(name, inp):
    #print list with 4 decimal for each item
    print(name, [round(x, 4) for x in inp])

from torchmetrics.classification import MulticlassHingeLoss
from torchmetrics.utilities.data import to_onehot
from torchmetrics.metric import Metric
from torchmetrics.functional.classification.confusion_matrix import _multiclass_confusion_matrix_format
from torchmetrics.functional.classification.hinge import (
    _multiclass_hinge_loss_arg_validation, 
    _multiclass_hinge_loss_tensor_validation,
    _hinge_loss_compute
)

def _custom_multiclass_hinge_loss_update(
    preds,
    target,
    alpha,
    squared,
    multiclass_mode = "crammer-singer"
):
    if not torch.all((preds >= 0) * (preds <= 1)):
        preds = preds.softmax(1)

    target = to_onehot(target, max(2, preds.shape[1])).bool()

    if multiclass_mode == "crammer-singer":
        margin = preds[target]

        # ── Guard: preds[~target] có thể rỗng khi batch chỉ có 1 class ──────
        # Xảy ra khi tất cả tokens trong batch thuộc cùng 1 class (vocab token)
        # → ~target mask toàn False → tensor 0 phần tử → reshape lỗi
        non_target_preds = preds[~target]
        if non_target_preds.numel() == 0:
            # Không có class nào khác để so sánh → loss = 0, bỏ qua batch này
            total = tensor(target.shape[0], device=target.device)
            zero = torch.zeros(preds.shape[1], device=preds.device, dtype=preds.dtype)
            return zero, total

        margin -= torch.max(non_target_preds.view(preds.shape[0], -1), dim=1)[0]
    else:
        target = target.bool()
        margin = torch.zeros_like(preds)
        margin[target] = preds[target]
        margin[~target] = -preds[~target]

    measures = alpha + margin
    measures = torch.clamp(measures, 0)

    if squared:
        measures = measures.pow(2)

    total = tensor(target.shape[0], device=target.device)
    return measures.sum(dim=0), total

def multiclass_hinge_loss(
    preds,
    target,
    num_classes,
    alpha = 1.0,
    squared = False,
    multiclass_mode = "crammer-singer",
    ignore_index = None,
    validate_args = True,
):
    if validate_args:
        _multiclass_hinge_loss_arg_validation(num_classes, squared, multiclass_mode, ignore_index)
        _multiclass_hinge_loss_tensor_validation(preds, target, num_classes, ignore_index)
    preds, target = _multiclass_confusion_matrix_format(preds, target, ignore_index, convert_to_labels=False)
    measures, total = _custom_multiclass_hinge_loss_update(
        preds, 
        target, 
        alpha,
        squared, 
        multiclass_mode,
    )
    return _hinge_loss_compute(measures, total)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids, labels, attention_mask = inputs
        # forward pass
        outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)

class CustomTrainerForgetting(Trainer):
    def __init__(self, *args, **kwargs):
        self.loss_type = kwargs.pop('forget_loss')
        self.oracle_model = kwargs.pop('oracle_model')
        self.eval_cfg = kwargs.pop('eval_cfg')

        # Parameters added for NPO
        self.seed = kwargs.pop('seed')
        self.npo_coeff=kwargs.pop('npo_coeff')
        self.grad_diff_coeff=kwargs.pop('grad_diff_coeff')
        self.KL_coeff=kwargs.pop('KL_coeff')
        self.ref_policy = kwargs.pop('ref_policy')
        self.beta = kwargs.pop('beta')

        #Add for SimNPO
        self.gamma = kwargs.pop('gamma',0.0)
        self.control_alpha = kwargs.pop('control_alpha', 1.0)
        self.control_lambda = kwargs.pop('control_lambda', 0.1)
        self.select_gamma = kwargs.pop('select_gamma', 0.1)
        self.prod_settings = kwargs.pop('prod_settings', {})

        super(CustomTrainerForgetting, self).__init__(*args, **kwargs)
        if self.loss_type in ('codeeraser', 'prod'):
            # Our loss is already a mean; Trainer must divide it by accumulation steps.
            self.model_accepts_loss_kwargs = False
        if self.loss_type in ("KL", "codeeraser", "prod") or ("npo" in self.loss_type and "simnpo" not in self.loss_type):
            self.oracle_model = self.e_prepare_deepspeed(self.oracle_model)

    def _get_train_sampler(self, train_dataset=None):
        if self.loss_type == 'codeeraser':
            dataset = train_dataset if train_dataset is not None else self.train_dataset
            effective_batch = (self.args.per_device_train_batch_size
                               * self.args.gradient_accumulation_steps
                               * self.accelerator.num_processes)
            return AlternatingBatchSampler(len(dataset), effective_batch, self.seed)
        return super()._get_train_sampler(train_dataset)

    def e_prepare_deepspeed(self, model):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = copy.deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        config_kwargs["optimizer"] = {"type": None}
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        if self.loss_type == 'codeeraser':
            # global_step changes only after a complete accumulation window.
            forgetting = self.state.global_step % 2 == 0
            batch = inputs['forget' if forgetting else 'retain']
            model_inputs = {key: batch[key] for key in ('input_ids', 'attention_mask')}
            outputs = model(**model_inputs)
            with torch.no_grad():
                reference = self.oracle_model(**model_inputs)
            if forgetting:
                loss = forget_objective(outputs.logits, reference.logits, batch['labels'],
                                        batch['secret_mask'], self.control_lambda,
                                        self.select_gamma)
            else:
                loss = retain_objective(outputs.logits, reference.logits, batch['labels'],
                                        self.control_alpha, self.control_lambda)

        elif self.loss_type == 'prod':
            model_inputs = {key: inputs[key] for key in ('input_ids', 'attention_mask')}
            outputs = model(**model_inputs)
            with torch.no_grad():
                reference = self.oracle_model(**model_inputs)
            loss = prod_objective(outputs.logits, reference.logits, inputs['labels'],
                                  **self.prod_settings)
            if model.training:
                # Upstream sums microbatch gradients. Cancel Trainer's averaging.
                loss = loss * getattr(self, 'current_gradient_accumulation_steps',
                                      self.args.gradient_accumulation_steps)

        elif self.loss_type == "grad_ascent":
            
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1
            loss = forget_loss

        elif self.loss_type == "grad_diff":
            
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            loss = forget_loss + retain_loss

        elif self.loss_type == "IHL":
            
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)

            scores = outputs.logits
            shift_logits = scores[..., :-1, :].contiguous().squeeze().view(-1, scores.size(-1))  # [BN, V]
            shift_labels = labels[..., 1:].contiguous().squeeze().view(-1)                        # [BN,]

            # Filter out padding tokens
            valid_mask  = shift_labels != -100
            valid_logits = shift_logits[valid_mask]
            valid_labels = shift_labels[valid_mask]

            if valid_logits.shape[0] == 0:
                # Toàn bộ batch là padding — skip, trả về 0 loss
                forget_loss = torch.tensor(0.0, requires_grad=True, device=scores.device)
            else:
                forget_loss = multiclass_hinge_loss(
                    valid_logits,
                    valid_labels,
                    valid_logits.size(-1),
                )

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            
            loss = forget_loss + retain_loss
        
        elif self.loss_type == "KL":
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            with torch.no_grad():
                retain_outputs = self.oracle_model(
                    retain_input_ids,labels=retain_labels, 
                    attention_mask=retain_attention_mask
                )
            
            retain_probs = F.log_softmax(retain_outputs.logits, dim=-1)
            retain_probs = retain_probs.view(-1, retain_outputs.logits.shape[-1])

            current_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            current_probs = F.log_softmax(current_outputs.logits, dim=-1)
            current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])

            retain_loss = nn.functional.kl_div(current_probs, retain_probs, reduction='batchmean', log_target=True)
            loss = forget_loss + retain_loss

        elif self.loss_type == "idk":
            idk_inputs, retain_inputs = inputs
            idk_input_ids, idk_labels, idk_attention_mask = idk_inputs
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            
            input_ids = torch.cat((idk_input_ids, retain_input_ids), dim=0)
            labels = torch.cat((idk_labels, retain_labels), dim=0)
            attention_mask = torch.cat((idk_attention_mask, retain_attention_mask), dim=0)
            
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            loss = outputs.loss
        
        elif self.loss_type == "dpo":
            idk_inputs, forget_inputs, retain_inputs = inputs
            idk_input_ids, idk_labels, idk_attention_mask = idk_inputs
            forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
            idk_outputs = model(idk_input_ids,labels=idk_labels, attention_mask=idk_attention_mask)
            forget_outputs = model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask)

            with torch.no_grad():
                idk_outputs_oracle = self.oracle_model(
                    idk_input_ids,
                    labels=idk_labels, 
                    attention_mask=idk_attention_mask
                )
                forget_outputs_oracle = self.oracle_model(
                    forget_input_ids,
                    labels=forget_labels, 
                    attention_mask=forget_attention_mask
                )
                idk_logits_oracle = idk_outputs_oracle.logits
                forget_logits_oracle = forget_outputs_oracle.logits

            idk_loss_oracle = -1 * get_batch_loss(idk_logits_oracle, idk_labels)
            forget_loss_oracle = -1 * get_batch_loss(forget_logits_oracle, labels)
            
            idk_loss_current = -1 * get_batch_loss(idk_outputs.logits, idk_labels)
            forget_loss_current = -1 * get_batch_loss(forget_outputs.logits, labels)

            pi_logratios = idk_loss_current - forget_loss_current
            ref_logratios = idk_loss_oracle - forget_loss_oracle

            beta = 0.1
            loss = -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()
            loss = -pi_logratios.mean()
            loss = -idk_loss_current.mean()
            outputs = forget_outputs
            
        elif self.loss_type == 'npo_grad_diff':
            
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            forget_loss_current = get_batch_loss(outputs.logits, labels) 

            if self.ref_policy == 'fine_tuned':
                with torch.no_grad():
                    forget_outputs_oracle = self.oracle_model(
                        input_ids,labels=labels, 
                        attention_mask=attention_mask
                    )
                    forget_logits_oracle = forget_outputs_oracle.logits
                    forget_loss_oracle = get_batch_loss(forget_logits_oracle, labels)
                neg_log_ratios = forget_loss_current - forget_loss_oracle
            else:
                raise NotImplementedError
            forget_loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            loss = self.npo_coeff * forget_loss + self.grad_diff_coeff * retain_loss

        elif self.loss_type == 'simnpo':
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)

            forget_loss_current = get_batch_loss(outputs.logits, labels)
            seq_len = (labels != -100).sum(dim=-1).float().clamp(min=1)
            neg_log_ratio = forget_loss_current / seq_len - self.gamma
            loss = -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta

        elif self.loss_type == 'simnpo_gdr':
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)

            forget_loss_current = get_batch_loss(outputs.logits, labels)
            seq_len = (labels != -100).sum(dim=-1).float().clamp(min=1)
            neg_log_ratio = forget_loss_current / seq_len - self.gamma
            forget_loss = -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(
                retain_input_ids, labels=retain_labels,
                attention_mask=retain_attention_mask
            )
            retain_loss = retain_outputs.loss
            loss = self.npo_coeff * forget_loss + self.grad_diff_coeff * retain_loss
        
        return (loss, outputs) if return_outputs else loss

        
        
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)

    def evaluate(
        self,
        eval_dataset = None,
        ignore_keys = None,
        metric_key_prefix = "eval",
    ):
        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)
        args = self.args
        model = self._wrap_model(self.model, training=False, dataloader=None)
        print(
            self.is_in_train, 
            args.device, 
            model.dtype, 
            self.args.dataloader_num_workers, 
            self.eval_cfg.split_list, 
            self.eval_cfg.split
        )
        
        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            if model is not self.model:
                self.model_wrapped = model

            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)
        model.eval()
        curr_step = self.state.global_step
        eval_cfg = self.eval_cfg

        curr_save_dir = os.path.join(eval_cfg.save_dir, f"checkpoint-{curr_step}")
        Path(curr_save_dir).mkdir(parents=True, exist_ok=True)
        forget_rate = eval_cfg.split.split('_')[0]
        with torch.no_grad():
            for i, (
                folder, 
                split, 
                question_key, 
                answer_key, 
                eval_task, 
                base_answer_key, 
                perturbed_answer_key
            ) in enumerate(zip(
                eval_cfg.data_path, 
                eval_cfg.split_list, 
                eval_cfg.question_key, 
                eval_cfg.answer_key, 
                eval_cfg.eval_task, 
                eval_cfg.base_answer_key, 
                eval_cfg.perturbed_answer_key
            )):
                
                world_size = self.accelerator.num_processes

                if eval_task == 'eval_log_forget':
                    split = eval_cfg.split
                print(f'Working on eval task {eval_task} with split {split}')
                save_filename = os.path.join(curr_save_dir, f"{eval_task}.json")
                save_filename = save_filename if world_size == 1 else os.path.join(
                    curr_save_dir, 
                    f"{eval_task}_{self.accelerator.local_process_index}.json"
                )
                if os.path.exists(save_filename) and not eval_cfg.overwrite:
                    print(f"Skipping {eval_task} because {save_filename} already exists")
                    continue

                eval_dataloader, base_eval_dataloader, perturb_dataloader = get_dataloader(
                    eval_cfg, 
                    eval_task, 
                    self.tokenizer, 
                    folder, 
                    split, 
                    question_key, 
                    answer_key, 
                    base_answer_key, 
                    perturbed_answer_key
                )
                eval_dataloader = self.accelerator.prepare(eval_dataloader)
                base_eval_dataloader = self.accelerator.prepare(base_eval_dataloader)
                perturb_dataloader = self.accelerator.prepare(perturb_dataloader)
                normalize_gt = False 

                eval_logs = get_all_evals(
                    eval_cfg, 
                    model, 
                    self.tokenizer, 
                    eval_task, 
                    eval_dataloader, 
                    base_eval_dataloader, 
                    perturb_dataloader, 
                    normalize_gt=normalize_gt
                )

                with open(save_filename, "w") as f:
                    json.dump(eval_logs, f, indent=4)
            
            self.accelerator.wait_for_everyone()
            aggregated_eval_logs = {}
            for eval_task in eval_cfg.eval_task:
                if world_size > 1:
                    if self.accelerator.is_local_main_process:
                        eval_logs = json.load(open(os.path.join(curr_save_dir, f"{eval_task}_0.json")))
                        for i in range(1, world_size):
                            filename = os.path.join(curr_save_dir, f"{eval_task}_{i}.json")
                            eval_logs = merge_dicts(eval_logs, json.load(open(filename)))
                        
                        aggregated_eval_logs[f'{eval_task}.json'] = eval_logs

                        new_save_filename = os.path.join(curr_save_dir, f"{eval_task}.json")
                        with open(new_save_filename, "w") as f:
                            json.dump(eval_logs, f, indent=4)

                            for i in range(world_size):
                                filename = os.path.join(curr_save_dir, f"{eval_task}_{i}.json")
                                os.remove(filename)
                                
            if self.accelerator.is_local_main_process:
                aggregated_eval_log_filename = os.path.join(curr_save_dir, "eval_log_aggregated.json")

                with open(aggregated_eval_log_filename, 'w') as f:
                    json.dump(aggregated_eval_logs, f, indent=4)

                if eval_cfg.retain_result is not None:
                    model_utility = get_model_utility(aggregated_eval_logs)
                    retain_result = json.load(open(eval_cfg.retain_result, 'r'))
                    forget_quality = get_forget_quality(aggregated_eval_logs, retain_result)
                    aggregate_stat = {**model_utility, **forget_quality}

                    with open(os.path.join(curr_save_dir, "aggregate_stat.csv"), 'w') as csvfile:
                        field_names = list(aggregate_stat.keys())
                        writer = csv.DictWriter(csvfile, fieldnames=field_names)
                        writer.writeheader()
                        writer.writerow(aggregate_stat)

def custom_data_collator_forget(samples):
    forget_samples, retain_samples = [sample[0] for sample in samples], [sample[1] for sample in samples]
    rets = []
    for data_type in ["forget", "retain"]:
        data = forget_samples if data_type == "forget" else retain_samples
        input_ids = [s[0] for s in data]
        labels = [s[1] for s in data]
        attention_mask = [s[2] for s in data]
        rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))
    return rets

def compute_metrics(pred):
    logits, labels = torch.from_numpy(pred.predictions), torch.from_numpy(pred.label_ids)
    preds = torch.from_numpy(pred.predictions.argmax(-1))
    shifted_labels = labels[..., 1:].contiguous()
    acc = torch.mean((preds[..., :-1] == shifted_labels).float())
    loss  = get_loss(logits, labels)
    return {"eval accuracy": acc, "eval loss": loss.item()}

def get_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_function(output.view(-1, output.size(-1)), shifted_labels.view(-1))

    return loss
