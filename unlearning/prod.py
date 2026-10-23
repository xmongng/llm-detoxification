"""PROD: redistribution targets from a frozen starting model.

Source: jiangxxxue/PROD, commit a581364, PROD.py (AAAI 2026).
The QA adapter selects answer tokens rather than the upstream fixed position
mask. Padding and ignored labels never enter the distribution or reduction.
"""
import json

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from codeeraser import encode_example


class PRODDataset(Dataset):
    """Forget-only JSONL training; optional retain records are for metrics only."""

    def __init__(self, forget_path, retain_path, tokenizer, model_config, max_length):
        if not tokenizer.is_fast:
            raise ValueError('PROD requires a fast tokenizer for answer boundaries.')
        self.tokenizer, self.model_config = tokenizer, model_config
        self.max_length = max_length
        self.forget = self._read(forget_path)
        self.retain = self._read(retain_path) if retain_path else []
        for index in range(len(self.forget)):
            try:
                self[index]
            except ValueError as exc:
                raise ValueError(f'PROD forget example {index}: {exc}') from exc

    @staticmethod
    def _read(path):
        with open(path, encoding='utf-8') as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        if not records:
            raise ValueError(f'Empty PROD dataset: {path}')
        for record in records:
            if not all(isinstance(record.get(k), str) for k in ('question', 'answer')):
                raise ValueError(f'{path}: expected string question and answer fields.')
        return records

    def __len__(self):
        return len(self.forget)

    def __getitem__(self, index):
        record = self.forget[index]
        # PROD targets the entire answer, independently of CodeEraser spans.
        encoded = encode_example(self.tokenizer, record['question'], record['answer'],
                                 self.model_config, self.max_length)
        encoded['labels'][~encoded.pop('secret_mask')] = -100
        return encoded


def prod_collator(samples):
    return {key: torch.stack([sample[key] for sample in samples]) for key in samples[0]}


@torch.no_grad()
def redistribution_targets(reference_logits, token_ids, top_p=0.8, alpha=0.0,
                           temperature=None, max_N=None):
    """Upstream deletion -> nucleus filtering -> softmax -> alpha suppression.

    Inputs contain only selected positions, [tokens, vocabulary]. With alpha>0
    the target has signed weights (Eq. 5); do not normalize it again.
    """
    if not 0 < top_p <= 1 or alpha < 0:
        raise ValueError('PROD requires 0 < top_p <= 1 and alpha >= 0.')
    if temperature is not None and temperature <= 0:
        raise ValueError('PROD temperature must be positive or null.')
    if max_N is not None and (type(max_N) is not int or max_N < 1):
        raise ValueError('PROD max_N must be a positive integer or null.')
    original = reference_logits.float()
    original_probs = F.softmax(original, dim=-1)
    deleted = original.clone().scatter_(-1, token_ids[:, None], -torch.inf)
    sorted_probs, indices = deleted.softmax(-1).sort(dim=-1, descending=True)
    remove = sorted_probs.cumsum(-1) > top_p
    remove[:, 1:] = remove[:, :-1].clone()
    remove[:, 0] = False
    if max_N is not None:
        remove |= torch.arange(original.shape[-1], device=original.device) >= max_N
    remove = torch.zeros_like(remove).scatter(-1, indices, remove)
    filtered = deleted.masked_fill(remove, -torch.inf)
    if temperature is not None:
        filtered = filtered / temperature
    target = filtered.softmax(-1)
    target.scatter_(-1, token_ids[:, None],
                    -alpha * original_probs.gather(-1, token_ids[:, None]))
    return target


def prod_objective(logits, reference_logits, labels, **settings):
    """Original soft-target CE, averaged over valid answer positions.

    Keep the upstream epsilon inside log; compute in float32 for bf16 safety.
    """
    valid = labels[:, 1:] != -100
    if not valid.any():
        raise ValueError('PROD batch has no answer tokens after truncation.')
    target = redistribution_targets(reference_logits[:, :-1][valid],
                                    labels[:, 1:][valid], **settings)
    probabilities = logits[:, :-1][valid].float().softmax(-1)
    return -(target * (probabilities + 1e-10).log()).sum(-1).mean()
