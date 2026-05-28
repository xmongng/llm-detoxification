import os
import json
from datasets import load_dataset


output_file = "python_code_18k.jsonl"
if os.path.exists("/kaggle/working"):
    output_file = "/kaggle/working/python_code_18k.jsonl"

print(f"Loading dataset...")
dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")

num_samples = 18000
print(f"Selecting {num_samples} samples...")
samples = dataset.shuffle(seed=42).select(range(num_samples))

with open(output_file, "w", encoding="utf-8") as f:
    for sample in samples:
        instruction = sample["instruction"]
        inp = sample.get("input", "").strip()
        
        # Merge input into question if present
        if inp and inp.lower() != "not applicable":
            question = f"{instruction}\n\nInput: {inp}"
        else:
            question = instruction
            
        record = {
            "question": question,
            "answer": sample["output"]
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Successfully saved {num_samples} samples to: {output_file}")
