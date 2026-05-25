# Detoxing LLMs: Unlearning to Filter Out Malicious Source Code

This repository provides a reproducible pipeline for collecting malicious-code prompts, evaluating jailbreak behavior, and applying unlearning methods to reduce unsafe code generation while preserving general coding ability.

## Repository Layout

```text
unlearning/
  config/                 # Hydra configs for training and evaluation
  datasets/               # Dataset preparation and malicious dataset collection tools
  evals/                  # Evaluation scripts
  data_module.py          # Local/Hugging Face dataset loaders
  forget.py               # Main unlearning entry point
  jailbreak.py            # Jailbreak generation entry point
bigcode-evaluation/       # Downstream coding benchmark harness
```

Most commands below should be run from the `unlearning/` directory unless stated otherwise.

---

## Step 1: Dataset Preparation

### 1.1 Malicious Dataset (Forget Set)

A prepared malicious-code dataset is included as:

```text
unlearning/datasets/malicious_code_origin.jsonl.zip
```

Extract it from the `unlearning/` directory:

```bash
unzip datasets/malicious_code_origin.jsonl.zip -d datasets/
```

The extracted file can be used as the initial malicious dataset:

```text
datasets/malicious_code_origin.jsonl
```

Original source: [pypi_malregistry](https://github.com/lxyeternal/pypi_malregistry)

### 1.2 Optional: Collect the Malicious Dataset Yourself

If you want to rebuild the malicious dataset from PyPI malware package archives, use the collection pipeline in `unlearning/datasets/collect_malicious.py`.

First, clone the source malware registry into the repository data directory:

```bash
cd /path/to/efficient-llm-unlearning
mkdir -p data
git clone https://github.com/lxyeternal/pypi_malregistry.git data/pypi_malregistry
```

Then run the full pipeline:

```bash
python -m unlearning.datasets.collect_malicious all \
  --repo-dir data/pypi_malregistry \
  --extracted-dir data/pypi_extracted \
  --raw-output data/malicious_snippets.jsonl \
  --clean-output data/malicious_clean.jsonl \
  --qa-output unlearning/datasets/malicious_code_origin.jsonl \
  --workers 4
```

The pipeline performs four stages:

1. `extract`: safely extracts `.tar.gz`, `.zip`, `.whl`, `.egg`, and related archives.
2. `scan`: scans Python files for suspicious code snippets using AST blocks and regex pattern groups.
3. `filter`: removes noisy, duplicated, too-short, too-long, or likely false-positive snippets.
4. `convert`: converts the cleaned snippets into the repository training format:

```json
{"question": "...", "answer": "..."}
```

You can also run each stage separately:

```bash
python -m unlearning.datasets.collect_malicious extract \
  --repo-dir data/pypi_malregistry \
  --out-dir data/pypi_extracted

python -m unlearning.datasets.collect_malicious scan \
  --extracted-dir data/pypi_extracted \
  --output data/malicious_snippets.jsonl \
  --workers 4

python -m unlearning.datasets.collect_malicious filter \
  --input data/malicious_snippets.jsonl \
  --output data/malicious_clean.jsonl

python -m unlearning.datasets.collect_malicious convert \
  --input data/malicious_clean.jsonl \
  --output unlearning/datasets/malicious_code_origin.jsonl \
  --cut-ratio 0.3
```

Useful options:

```bash
python -m unlearning.datasets.collect_malicious --help
python -m unlearning.datasets.collect_malicious scan --help
python -m unlearning.datasets.collect_malicious filter --help
python -m unlearning.datasets.collect_malicious convert --help
```

### 1.3 Normal Python Dataset (Retain Set)

To download and prepare the clean Python code retain dataset from Hugging Face, run:

```bash
cd unlearning
python datasets/prepare_data.py
```

This creates:

```text
python_code_18k.jsonl
```

If needed, move it into `unlearning/datasets/` or update the training command paths accordingly.

---

## Step 2: Jailbreak Attack

Run `jailbreak.py` with a system prompt and prefix injection to generate malicious completions from the target model.

```bash
cd unlearning
python jailbreak.py \
  model_family=deepseek-coder-6.7b \
  model_path=models/deepseek-coder-6.7b-instruct \
  data_path=datasets/malicious_code_origin.jsonl \
  output_path=datasets/predictions.jsonl
```

Default parameters can be edited in:

```text
unlearning/config/jailbreak.yaml
```

---

## Step 3: Maliciousness Evaluation

Evaluate the completions generated in Step 2 with the LLM-as-a-judge evaluator:

```bash
cd unlearning
python evals/eval_malicious.py
```

The output labels are used to identify samples where jailbreak succeeded.

---

## Step 4: Extract the Final Forget Set

From the labeled results in Step 3, keep examples with a `BAD` label. These examples represent prompts where the model generated malicious code and should be used as the final forget set.

Save the filtered data as a JSONL file with the same schema:

```json
{"question": "...", "answer": "..."}
```

Example path used by the commands below:

```text
unlearning/datasets/deepseek6.7b_bad.jsonl
```

---

## Step 5: Model Unlearning

Run `forget.py` to unlearn the malicious behavior using one of the supported objectives. The examples below use DeepSeek-Coder as an illustration; change `model_family`, `model_path`, and dataset paths for your own model.

### Option A: Importance-based Hashed Loss (IHL)

IHL uses LoRA for parameter-efficient unlearning.

```bash
cd unlearning
torchrun \
  --nproc_per_node=1 \
  --master_addr=localhost \
  --master_port=29500 \
  forget.py \
  --config-path ./config \
  --config-name forget \
  forget_data_path=datasets/deepseek6.7b_bad.jsonl \
  retain_data_path=datasets/python_code_18k.jsonl \
  model_family=deepseek-coder-6.7b \
  model_path=models/deepseek-coder-6.7b-instruct \
  forget_loss=IHL \
  importance_file=null \
  'LoRA.r=16' \
  'LoRA.alpha=32' \
  'LoRA.targets=all' \
  'LoRA.dropout=0.05' \
  num_epochs=5 \
  lr=1e-5 \
  batch_size=2 \
  gradient_accumulation_steps=16 \
  max_length=2048
```

### Option B: Gradient Difference

Gradient Difference maximizes loss on the forget set while minimizing loss on the retain set.

```bash
cd unlearning
torchrun \
  --nproc_per_node=1 \
  --master_addr=localhost \
  --master_port=29500 \
  forget.py \
  --config-path ./config \
  --config-name forget \
  forget_data_path=datasets/deepseek6.7b_bad.jsonl \
  retain_data_path=datasets/python_code_18k.jsonl \
  model_family=deepseek-coder-6.7b \
  model_path=models/deepseek-coder-6.7b-instruct \
  forget_loss=grad_diff \
  importance_file=null \
  lr=2e-5 \
  batch_size=2 \
  gradient_accumulation_steps=85 \
  num_epochs=1 \
  max_length=2048
```

### Option C: SimNPO with Gradient Difference Regularization

SimNPO with GDR optimizes a preference-style unlearning objective with retain-set regularization.

```bash
cd unlearning
torchrun \
  --nproc_per_node=1 \
  --master_addr=localhost \
  --master_port=29500 \
  forget.py \
  --config-path ./config \
  --config-name forget \
  forget_data_path=datasets/deepseek6.7b_bad.jsonl \
  retain_data_path=datasets/python_code_18k.jsonl \
  model_family=deepseek-coder-6.7b \
  model_path=models/deepseek-coder-6.7b-instruct \
  forget_loss=simnpo_gdr \
  importance_file=null \
  num_epochs=10 \
  lr=1e-5 \
  batch_size=2 \
  gradient_accumulation_steps=16 \
  beta=0.7 \
  grad_diff_coeff=0.1 \
  npo_coeff=1.0 \
  max_length=2048
```

LoRA is enabled for `IHL` and disabled for `grad_diff` and `simnpo_gdr`.

---

## Step 6: Post-Unlearning Evaluation

Evaluate whether the unlearned model reduces memorization of the forget set while preserving retain-set behavior:

```bash
cd unlearning
python evals/eval_memorize.py \
  model_family=deepseek-coder-6.7b \
  model_path=models/deepseek6.7b_unlearned \
  data_path=datasets/deepseek6.7b_bad.jsonl \
  output_dir=results \
  run_name=deepseek6.7b_unlearned \
  eval_tokens=100
```

The evaluator reports:

1. **MA (Memorization Accuracy)**: token-level reconstruction accuracy.
2. **EL (Extraction Likelihood)**: n-gram overlap recall on generated suffixes.
3. **Loss and PPL**: perplexity on forget and retain samples.

Outputs are saved as:

```text
{run_name}_per_example.csv
{run_name}_summary.json
```

---

## Step 7: Downstream Coding and Reasoning Evaluation

Use `bigcode-evaluation-harness` to test whether unlearning preserves general coding and reasoning ability.

### 7.1 Setup

```bash
cd bigcode-evaluation/bigcode-evaluation-harness
pip install -e .
```

### 7.2 HumanEval

```bash
accelerate launch main.py \
  --model models/deepseek6.7b_unlearned \
  --tasks instruct-humaneval \
  --instruction_tokens "<|im_start|>user\n,<|im_end|>\n,<|im_start|>assistant\n" \
  --max_length_generation 512 \
  --temperature 0.2 \
  --do_sample True \
  --n_samples 1 \
  --batch_size 1 \
  --precision fp16 \
  --allow_code_execution \
  --save_generations \
  --metric_output_path results_instruct.json
```

### 7.3 MBPP

```bash
accelerate launch main.py \
  --model models/deepseek6.7b_unlearned \
  --tasks mbpp \
  --max_length_generation 512 \
  --temperature 0.1 \
  --do_sample True \
  --n_samples 1 \
  --batch_size 1 \
  --precision fp16 \
  --allow_code_execution \
  --save_generations \
  --metric_output_path results_mbpp.json
```

### 7.4 GSM8K

```bash
accelerate launch main.py \
  --model models/deepseek6.7b_unlearned \
  --tasks pal-gsm8k-greedy \
  --max_length_generation 2048 \
  --n_samples 1 \
  --batch_size 1 \
  --do_sample False \
  --trust_remote_code \
  --precision fp16 \
  --allow_code_execution \
  --save_generations \
  --save_generations_path gsm8k_smoke.json \
  --metric_output_path gsm8k_smoke_results.json
```

---

## Notes

- Use relative paths where possible so experiments are reproducible across machines.
- Keep forget and retain datasets in the same `{question, answer}` JSONL format.
- Run collection and unlearning experiments on a small subset first before launching a full run.
