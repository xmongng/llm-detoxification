import json
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import datasets
from utils import get_model_identifiers_from_yaml, add_dataset_index


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_jsonl(path):
    """Load file JSONL local."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def convert_raw_data_to_model_format(tokenizer, max_length, question, answer, model_configs):
    question_start_token = model_configs['question_start_tag']
    question_end_token   = model_configs['question_end_tag']
    answer_token         = model_configs['answer_tag']

    new_question = question_start_token + question + question_end_token
    new_answer   = answer_token + answer
    full_text    = new_question + new_answer

    num_question_tokens = min(
        len(tokenizer.tokenize(new_question, add_special_tokens=True)),
        max_length
    )

    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )
    pad_length         = max_length - len(encoded.input_ids)
    pad_input_ids      = encoded['input_ids']      + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length

    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids[:]
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length - 1)

    # Mask question tokens
    for i in range(num_question_tokens):
        label[i] = -100

    return (
        torch.tensor(pad_input_ids),
        torch.tensor(label),
        torch.tensor(pad_attention_mask),
    )


# ── Datasets local JSONL ───────────────────────────────────────────────────

class LocalTextForgetDatasetQA(Dataset):
    """
    Load forget/retain từ 2 file JSONL local.
    Dùng cho: grad_diff | IHL | simnpo_gdr | grad_ascent | KL | npo_grad_diff
    Mỗi item trả về: [forget_tuple, retain_tuple]
    """
    def __init__(
        self,
        forget_data_path,
        retain_data_path,
        tokenizer,
        model_family,
        max_length=512,
        loss_type="grad_diff",
    ):
        super().__init__()
        self.tokenizer     = tokenizer
        self.max_length    = max_length
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.loss_type     = loss_type

        self.forget_data = load_jsonl(forget_data_path)
        self.retain_data = load_jsonl(retain_data_path)

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []

        # ── Forget sample ──────────────────────────────────────────────────
        f = self.forget_data[idx]
        forget_converted = convert_raw_data_to_model_format(
            self.tokenizer, self.max_length,
            f["question"], f["answer"],
            self.model_configs,
        )
        rets.append(forget_converted)

        # ── Retain sample  ─────────────
        retain_idx = (
            idx + torch.randint(0, len(self.retain_data), (1,)).item()
        ) % len(self.retain_data)
        r = self.retain_data[retain_idx]
        retain_converted = convert_raw_data_to_model_format(
            self.tokenizer, self.max_length,
            r["question"], r["answer"],
            self.model_configs,
        )
        rets.append(retain_converted)

        return rets  # [forget_tuple, retain_tuple]


class LocalTextForgetDatasetIDK(Dataset):
    """
    Load forget/retain từ 2 file JSONL local.
    Dùng cho: idk
    Mỗi item trả về: [idk_tuple, retain_tuple]
    idk_tuple: cùng question với forget nhưng answer thay bằng câu từ idontknow.jsonl
    """
    def __init__(
        self,
        forget_data_path,
        retain_data_path,
        tokenizer,
        model_family,
        max_length=512,
        idk_file="data/idontknow.jsonl",
    ):
        super().__init__()
        self.tokenizer     = tokenizer
        self.max_length    = max_length
        self.model_configs = get_model_identifiers_from_yaml(model_family)

        self.forget_data = load_jsonl(forget_data_path)
        self.retain_data = load_jsonl(retain_data_path)
        self.idk = open(idk_file, "r").readlines()

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []

        # ── IDK sample──────────────────
        f = self.forget_data[idx]
        rand_pos   = torch.randint(0, len(self.idk), (1,)).item()
        idk_answer = self.idk[rand_pos].strip()
        idk_converted = convert_raw_data_to_model_format(
            self.tokenizer, self.max_length,
            f["question"], idk_answer,
            self.model_configs,
        )
        rets.append(idk_converted)

        # ── Retain sample (random) ─────────────────────────────────────────
        retain_idx = (
            idx + torch.randint(0, len(self.retain_data), (1,)).item()
        ) % len(self.retain_data)
        r = self.retain_data[retain_idx]
        retain_converted = convert_raw_data_to_model_format(
            self.tokenizer, self.max_length,
            r["question"], r["answer"],
            self.model_configs,
        )
        rets.append(retain_converted)

        return rets  # [idk_tuple, retain_tuple]


# ── Datasets HuggingFace Hub ─────────────

class TextForgetDatasetQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split="forget10", loss_type="idk"):
        super(TextForgetDatasetQA, self).__init__()
        self.tokenizer     = tokenizer
        self.max_length    = max_length
        self.forget_data   = datasets.load_dataset(data_path, split)["train"]
        retain_split       = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
        self.retain_data   = datasets.load_dataset(data_path, retain_split)["train"]
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.loss_type     = loss_type

        if self.loss_type == "idk":
            self.split1, self.split2 = "idk", "retain"
            self.idontknowfile = "data/idontknow.jsonl"
            self.idk = open(self.idontknowfile, "r").readlines()
        else:
            self.split1, self.split2 = "forget", "retain"

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []
        for data_type in [self.split1, self.split2]:
            data = self.retain_data if data_type == "retain" else self.forget_data
            idx  = idx if data_type != "retain" else (
                idx + torch.randint(0, len(self.retain_data), (1,)).item()
            ) % len(self.retain_data)
            question = data[idx]['question']
            answer   = data[idx]['answer']

            if data_type == "idk":
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                answer   = self.idk[rand_pos].strip()

            converted_data = convert_raw_data_to_model_format(
                self.tokenizer, self.max_length,
                question, answer,
                self.model_configs,
            )
            rets.append(converted_data)
        return rets


class TextForgetDatasetDPOQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split="forget10"):
        super(TextForgetDatasetDPOQA, self).__init__()
        self.tokenizer     = tokenizer
        self.max_length    = max_length
        self.forget_data   = datasets.load_dataset(data_path, split)["train"]
        self.idontknowfile = "data/idontknow.jsonl"
        self.idk           = open(self.idontknowfile, "r").readlines()
        retain_split       = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
        self.retain_data   = datasets.load_dataset(data_path, retain_split)["train"]
        self.model_configs = get_model_identifiers_from_yaml(model_family)

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []
        for data_type in ["idk", "forget", "retain"]:
            data = self.forget_data if data_type != "retain" else self.retain_data
            idx  = idx if data_type != "retain" else (
                idx + torch.randint(0, len(self.retain_data), (1,)).item()
            ) % len(self.retain_data)
            question = data[idx]['question']
            if data_type != "idk":
                answer = data[idx]['answer']
            else:
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                answer   = self.idk[rand_pos].strip()

            converted_data = convert_raw_data_to_model_format(
                self.tokenizer, self.max_length,
                question, answer,
                self.model_configs,
            )
            rets.append(converted_data)
        return rets


class TextDatasetQA(Dataset):
    def __init__(
        self,
        data_path,
        tokenizer,
        model_family,
        max_length=512,
        split=None,
        question_key='question',
        answer_key='answer',
    ):
        super(TextDatasetQA, self).__init__()
        self.tokenizer     = tokenizer
        self.max_length    = max_length
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.qk = question_key
        self.ak = answer_key

        import os
        if os.path.isfile(data_path):
            # Local JSONL
            self.data = datasets.load_dataset("json", data_files=data_path, split="train")
        else:
            # HuggingFace Hub
            self.data = datasets.load_dataset(data_path, split)["train"]

        self.data = add_dataset_index(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers  = self.data[idx][self.ak]
        indices  = self.data[idx]['index']

        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list      = []
        label_list              = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_format(
                self.tokenizer, self.max_length,
                question, answer,
                self.model_configs,
            )
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])

        return (
            torch.stack(pad_input_ids_list).squeeze(),
            torch.stack(label_list).squeeze(),
            torch.stack(pad_attention_mask_list).squeeze(),
            torch.tensor(indices),
        )


# ── Collators ──────────────────────────────────────────────────────────────────

def collate_fn(batch):
    input_ids, attention_masks = zip(*batch)
    input_ids       = pad_sequence(input_ids,       batch_first=True, padding_value=-100)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    return input_ids, attention_masks


def custom_data_collator(samples):
    input_ids      = [s[0] for s in samples]
    labels         = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)


def custom_data_collator_with_indices(samples):
    input_ids      = [s[0] for s in samples]
    labels         = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    indices        = [s[3] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask), torch.stack(indices)


def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output         = output[..., :-1, :].contiguous()
    loss_function  = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    loss = loss_function(output.transpose(-1, -2), shifted_labels).sum(dim=-1)
    return loss