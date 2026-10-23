"""CodeEraser (SSCU): selective CE and contrastive reference KL.

The target is the answer by default. Optional ``forget_spans`` are half-open
character offsets [start, end] relative to the answer, for finer selection.
"""
import json
import math

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler


def encode_example(tokenizer, question, answer, model_config, max_length,
                   forget_spans=None, require_target=True):
    prefix = (model_config['question_start_tag'] + question
              + model_config['question_end_tag'] + model_config['answer_tag'])
    spans = ([[0, len(answer)]] if answer else []) if forget_spans is None else forget_spans
    for span in spans:
        if (len(span) != 2 or any(type(x) is not int for x in span)
                or not 0 <= span[0] < span[1] <= len(answer)):
            raise ValueError('forget_spans must contain valid [start, end] answer character offsets.')
    encoded = tokenizer(prefix + answer, max_length=max_length, truncation=True,
                        padding='max_length', return_offsets_mapping=True,
                        return_tensors='pt')
    ids = encoded['input_ids'][0]
    attention = encoded['attention_mask'][0]
    offsets = encoded['offset_mapping'][0]
    valid = attention.bool() & (offsets[:, 1] > offsets[:, 0])
    target = torch.zeros_like(valid)
    for start, end in spans:
        target |= ((offsets[:, 0] < len(prefix) + end)
                   & (offsets[:, 1] > len(prefix) + start) & valid)
    labels = ids.clone()
    labels[~valid] = -100
    if require_target and not target[1:].any():
        raise ValueError('No target tokens remain after truncation; increase max_length or fix forget_spans.')
    return {'input_ids': ids, 'attention_mask': attention, 'labels': labels,
            'secret_mask': target}


class CodeEraserDataset(Dataset):
    def __init__(self, forget_path, retain_path, tokenizer, model_config, max_length,
                 retain_max_length=128):
        if not tokenizer.is_fast:
            raise ValueError('CodeEraser requires a fast tokenizer with character offsets.')
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.max_length = max_length
        self.retain_max_length = retain_max_length
        self.forget = self._read(forget_path)
        self.retain = self._read(retain_path)
        # Check every forget example before expensive model loading/training.
        for index, record in enumerate(self.forget):
            try:
                self._encode(record, True)
            except ValueError as exc:
                raise ValueError(f'Forget example {index}: {exc}') from exc

    @staticmethod
    def _read(path):
        with open(path, encoding='utf-8') as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        if not records:
            raise ValueError(f'Empty CodeEraser dataset: {path}')
        for record in records:
            if not all(isinstance(record.get(key), str) for key in ('question', 'answer')):
                raise ValueError(f'{path}: each record needs string question and answer fields.')
        return records

    def _encode(self, record, is_forget):
        return encode_example(self.tokenizer, record['question'], record['answer'],
                              self.model_config, self.max_length if is_forget else self.retain_max_length,
                              record.get('forget_spans'), require_target=is_forget)

    def __len__(self):
        return len(self.forget)

    def __getitem__(self, index):
        retain_index = torch.randint(len(self.retain), ()).item()
        return {'forget': self._encode(self.forget[index], True),
                'retain': self._encode(self.retain[retain_index], False)}


def codeeraser_collator(samples):
    return {split: {key: torch.stack([sample[split][key] for sample in samples])
                    for key in samples[0][split]}
            for split in ('forget', 'retain')}


class AlternatingBatchSampler(Sampler):
    """Repeat each global effective batch for a forget then retain update.

    Trainer/Accelerate shards this index stream across ranks. Padding the final
    block makes every optimizer update contain a complete accumulation window.
    """
    def __init__(self, size, effective_batch_size, seed=42):
        self.size = size
        self.block_size = effective_batch_size
        self.seed = seed
        self.epoch = 0
        if size <= 0 or effective_batch_size <= 0:
            raise ValueError('Dataset and effective batch sizes must be positive.')

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return 2 * math.ceil(self.size / self.block_size) * self.block_size

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.size, generator=generator).tolist()
        padded_size = len(self) // 2
        indices = (indices * math.ceil(padded_size / self.size))[:padded_size]
        for start in range(0, padded_size, self.block_size):
            block = indices[start:start + self.block_size]
            yield from block
            yield from block


def selected_kl(logits, reference_logits, mask, reduction='tokenmean'):
    """KL(reference || current), with SSCU's per-target/per-batch reductions."""
    current = logits[:, :-1][mask].float()
    reference = reference_logits[:, :-1][mask].float()
    if not current.numel():
        return logits.sum() * 0
    # log_softmax avoids softmax-underflow followed by log in mixed precision.
    divergence = F.kl_div(F.log_softmax(current, dim=-1),
                          F.log_softmax(reference, dim=-1),
                          log_target=True, reduction='sum')
    denominator = current.shape[0] if reduction == 'tokenmean' else logits.shape[0]
    return divergence / denominator


def forget_objective(logits, reference_logits, labels, secret_mask,
                     control_lambda=0.1, select_gamma=0.1):
    shifted_labels = labels[:, 1:]
    valid = shifted_labels != -100
    secret = secret_mask[:, 1:].bool() & valid
    context = ~secret_mask[:, 1:].bool() & valid
    if not secret.any():
        raise ValueError('CodeEraser forget batch has no target tokens.')
    token_losses = F.cross_entropy(logits[:, :-1].float().transpose(1, 2),
                                   shifted_labels, reduction='none', ignore_index=-100)
    secret_loss = token_losses[secret].mean()
    context_loss = token_losses[context].mean() if context.any() else logits.sum() * 0
    kl = selected_kl(logits, reference_logits, secret)
    return -secret_loss + select_gamma * context_loss - control_lambda * kl


def retain_objective(logits, reference_logits, labels, control_alpha=1.0,
                     control_lambda=0.1):
    valid = labels[:, 1:] != -100
    if not valid.any():
        raise ValueError('CodeEraser retain batch has no valid tokens.')
    return control_lambda * control_alpha * selected_kl(
        logits, reference_logits, valid, reduction='batchmean')
