## Step-by-Step Guide

### Step 1: Dataset Preparation

1. **Malicious Dataset (Forget set)**: Use the raw malicious code dataset (`malicious_code_origin.jsonl`) located inside the `unlearning/datasets/` directory.
   * **Source/Original Data**: [pypi_malregistry](https://github.com/lxyeternal/pypi_malregistry)
   * The dataset is provided as a zip file `unlearning/datasets/malicious_code_origin.jsonl.zip`. Extract it by running the following command from the `unlearning/` directory:
     ```bash
     unzip datasets/malicious_code_origin.jsonl.zip -d datasets/
     ```

2. **Normal Python Dataset (Retain set)**: To download and prepare the clean Python code dataset (`python_code_18k.jsonl` from Hugging Face), run the following data preparation script:
   ```bash
   python datasets/prepare_data.py
   ```
   *This script automatically checks the environment, downloads the Python code instructions dataset from Hugging Face, shuffles it, and saves 18,000 samples to the configured output path.*

---

### Step 2: Jailbreak Attack (Generating Malicious Completions)
Run [jailbreak.py](unlearning/jailbreak.py) combined with a *System Prompt* and *Prefix Injection* to force the target model to generate malicious code completions.

**Run command:**
```bash
python jailbreak.py \
  model_family=deepseek-coder-6.7b \
  model_path=models/deepseek-coder-6.7b-instruct \
  data_path=datasets/malicious_code_origin.jsonl \
  output_path=datasets/predictions.jsonl
```
*Note: You can configure default parameters in [jailbreak.yaml](unlearning/config/jailbreak.yaml).*

---

### Step 3: Maliciousness Evaluation (LLM-as-a-Judge)
Evaluate the completions generated in Step 2 by running the evaluation script to classify responses.

**Run command:**
```bash
python evals/eval_malicious.py
```

---

### Step 4: Extracting the Forget Set
From the labeled results in Step 3, filter out all examples with a `BAD` label. This represents the subset of inputs where the model generated malicious code (jailbreak succeeded).
* Reformat this subset as a JSONL file to serve as the **Forget Dataset**.

---

### Step 5: Model Unlearning (Detoxification)
Run the [forget.py](unlearning/forget.py) script to unlearn the malicious dataset collected in Step 4. Below are the commands to execute the three supported algorithms. Note that these configurations can be customized for different models by adjusting the `model_family` and `model_path` arguments (the commands below use a sample model configuration for illustration):

#### Option A: Importance-based Hashed Loss (IHL)
IHL uses Parameter-Efficient Fine-Tuning (LoRA) to quickly unlearn targeted concepts.
```bash
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
  num_epochs=2 \
  lr=2e-5 \
  batch_size=1 \
  gradient_accumulation_steps=16 \
  max_length=512
```

#### Option B: Gradient Difference (grad_diff)
Gradient Difference performs full parameter tuning (without LoRA) to maximize the loss of the forget dataset while minimizing the loss of the retain dataset.
```bash
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
  batch_size=1 \
  gradient_accumulation_steps=8 \
  num_epochs=1 \
  max_length=512
```

#### Option C: SimNPO with Gradient Difference Regularization (simnpo_gdr)
SimNPO with GDR performs full parameter tuning (without LoRA) to optimize the unlearning objective.
```bash
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
  num_epochs=1 \
  lr=2e-5 \
  batch_size=1 \
  gradient_accumulation_steps=8 \
  beta=0.7 \
  grad_diff_coeff=0.1 \
  npo_coeff=1.0 \
  max_length=512
```

*Note: LoRA is auto-enabled only for the `IHL` loss and automatically disabled during `grad_diff` and `simnpo_gdr` training as per scientific standards.*

---

### Step 6: Post-Unlearning Evaluation
To verify that the model has successfully unlearned the target behavior without losing general coding capabilities, run the [eval_memorize.py](unlearning/evals/eval_memorize.py) evaluation script.

The script calculates:
1. **MA (Memorization Accuracy)**: Autoregressively measures the token-by-token reconstruction rate on the forget/jailbreak targets.
2. **EL (Extraction Likelihood)**: Computes n-gram overlap recall on generated suffixes.
3. **Loss & PPL (Perplexity)**: Measures model perplexity on both forget and retain datasets.

**Run command:**
```bash
python evals/eval_memorize.py \
  model_family=deepseek-coder-6.7b \
  model_path=models/deepseek6.7b_unlearned \
  data_path=datasets/deepseek6.7b_bad.jsonl \
  output_dir=results \
  run_name=deepseek6.7b_unlearned \
  eval_tokens=100
```
*Results will be saved to `{run_name}_per_example.csv` (per-example metrics) and `{run_name}_summary.json` (mean & standard deviation summary).*

---

### Step 7: Downstream Coding & Reasoning Capabilities (HumanEval, MBPP, GSM8K)
To evaluate the downstream coding capabilities and mathematical reasoning of the unlearned model on benchmark datasets, use the `bigcode-evaluation-harness`.

1. **Setup bigcode-evaluation-harness**:
   ```bash
   cd bigcode-evaluation/bigcode-evaluation-harness
   pip install -e .
   ```

2. **Evaluate HumanEval (Instruction Tuning)**:
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

3. **Evaluate MBPP**:
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

4. **Evaluate GSM8K (Math Reasoning)**:
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
