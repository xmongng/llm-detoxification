# LLM Detoxification — chạy trên Kaggle

Luồng: **dữ liệu gốc → jailbreak → judge → lọc BAD → unlearning → đánh giá lại**.
Copy mỗi khối code bên dưới vào một cell Python, chạy theo thứ tự.

Bật **GPU** và **Internet** trong Notebook Settings. Thêm Kaggle Secret
`JUDGE_API_KEY` cho dịch vụ judge tương thích Chat Completions. Judge gửi prompt
và câu trả lời tới dịch vụ đó và có thể phát sinh phí.
Mặc định dùng Qwen2.5-Coder-1.5B-Instruct và PROD + LoRA cho GPU nhỏ;
đây là biến thể LoRA, không phải cấu hình full-weight của bài báo.

## Cell 1 — tải code và cài thư viện

```python
import os, sys, subprocess
from pathlib import Path

REPO = Path('/kaggle/working/llm-detoxification')
if not REPO.exists():
    subprocess.run(['git', 'clone', 'https://github.com/xmongng/llm-detoxification.git', str(REPO)], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r',
                str(REPO / 'unlearning/requirements.txt')], check=True)
os.chdir(REPO / 'unlearning')
```

Nếu notebook đã import torch/transformers trước khi cài, restart session rồi
chạy lại các cell. Không cần cài FlashAttention.

## Cell 2 — cấu hình chung

Chỉ sửa các biến ở cell này. Khi tiếp tục phiên cũ, đặt `RUN` về đúng thư mục cũ.

```python
import json, torch
from datetime import datetime, timezone
from huggingface_hub import snapshot_download
from kaggle_secrets import UserSecretsClient

MODEL_FAMILY = 'qwen2.5-coder-1.5b'
MODEL_ID = 'Qwen/Qwen2.5-Coder-1.5B-Instruct'
MODEL_PATH = snapshot_download(MODEL_ID)  # Hoặc đường dẫn model đã gắn trong /kaggle/input/...
METHOD = 'prod'  # prod | codeeraser | grad_diff | simnpo | simnpo_gdr | IHL
USE_LORA = True  # False: full weights, cần nhiều VRAM hơn
JUDGE_MODEL = 'gpt-4'  # Đổi thành model judge mà tài khoản của bạn có quyền sử dụng
JUDGE_URL = 'https://api.openai.com/v1/chat/completions'
os.environ['JUDGE_API_KEY'] = UserSecretsClient().get_secret('JUDGE_API_KEY')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

assert torch.cuda.is_available(), 'Hãy bật GPU trong Notebook Settings.'
DTYPE = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
JAILBREAK_GPUS = torch.cuda.device_count()  # Kaggle T4 x2: dùng cả 2 GPU
JAILBREAK_BATCH = 4  # Mỗi GPU; giảm còn 2 hoặc 1 nếu hết VRAM
print('GPU inference:', JAILBREAK_GPUS, '— batch/GPU:', JAILBREAK_BATCH)
RUN = Path('/kaggle/working/experiments') / datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
RUN.mkdir(parents=True, exist_ok=True)
print('Lưu lại đường dẫn này:', RUN)

def run_script(script, *args):
    subprocess.run([sys.executable, script, *map(str, args)], check=True)

(RUN / 'settings.json').write_text(json.dumps(dict(
    model_family=MODEL_FAMILY, model_id=MODEL_ID, model_path=MODEL_PATH,
    method=METHOD, lora=USE_LORA, judge_model=JUDGE_MODEL, judge_url=JUDGE_URL,
    dtype=DTYPE, generation_limit=1024,
    jailbreak_gpus=JAILBREAK_GPUS, jailbreak_batch=JAILBREAK_BATCH), indent=2))
```

Để dùng DeepSeek 6.7B, đổi `MODEL_FAMILY='deepseek-coder-6.7b'` và
`MODEL_ID='deepseek-ai/deepseek-coder-6.7b-instruct'`. Model 6.7B cùng reference
có thể không vừa VRAM Kaggle, kể cả dùng LoRA. Jailbreak tự dùng mọi GPU nhìn thấy;
cell unlearning vẫn chạy **một GPU** để giữ cấu hình training hiện tại.

## Cell 3 — chuẩn bị dữ liệu gốc và retain

Dữ liệu gốc có sẵn trong repo (1.283 mẫu). Retain là code sạch, không phải kết quả jailbreak.

```python
import zipfile, shutil

SOURCE = RUN / 'malicious_original.jsonl'
with zipfile.ZipFile('datasets/malicious_code_origin.jsonl.zip') as archive:
    SOURCE.write_bytes(archive.read('malicious_code_origin.jsonl'))
RETAIN = RUN / 'retain.jsonl'
run_script('datasets/prepare_data.py')
shutil.copy2('/kaggle/working/python_code_18k.jsonl', RETAIN)
print('Original:', sum(1 for _ in SOURCE.open()))
print('Retain:', sum(1 for _ in RETAIN.open()))
```

## Cell 4 — jailbreak model ban đầu

```python
BEFORE = RUN / 'jailbreak_before.jsonl'
run_script('jailbreak.py', f'model_family={MODEL_FAMILY}', f'model_path={MODEL_PATH}',
           f'data_path={SOURCE}', f'output_path={BEFORE}',
           f'torch_dtype={DTYPE}', f'num_gpus={JAILBREAK_GPUS}', f'batch_size={JAILBREAK_BATCH}',
           'max_new_tokens=1024', 'temperature=0.1', 'seed=42')
```

Trên T4 x2, mỗi GPU giữ một bản model 1.5B ở FP16 và sinh batch riêng.
File `.jsonl.shards/` giữ kết quả từng GPU; file JSONL cuối chỉ được xuất khi
đã đủ mẫu, đúng thứ tự. File `.jsonl.summary.json` lưu tổng thời gian,
samples/giây và peak VRAM từng GPU. Không ghi đè kết quả cũ.
Nếu generation bị ngắt hoặc hết VRAM, giữ shards để kiểm tra, giảm batch nếu
cần và dùng output path mới để chạy lại. Giữ **cùng số GPU, batch và seed** ở
hai lần trước/sau: batching thay đổi luồng sampling so với bản chạy từng mẫu.

## Cell 5 — judge và tạo forget set

Judge tiếp tục từ những dòng đã lưu khi chạy lại cùng lệnh. Forget set dùng
**`predicted_answer` được gắn BAD**, không dùng đáp án gốc thay thế.

```python
LABELS_BEFORE = RUN / 'labels_before.jsonl'
FORGET = RUN / 'forget.jsonl'
run_script('evals/judge_jsonl.py', '--input', BEFORE, '--output', LABELS_BEFORE,
           '--forget-output', FORGET, '--model', JUDGE_MODEL, '--url', JUDGE_URL)
print('Forget:', sum(1 for _ in FORGET.open()))
```

## Cell 6 — chạy unlearning

Mỗi lần chạy tạo thư mục kết quả mới. `prod` và `codeeraser` chọn cấu hình riêng
để giữ hyperparameter từ repo gốc; các method còn lại dùng `forget.yaml`.
PROD chỉ học trên forget set, retain chỉ dùng cho metrics.

```python
CONFIG = METHOD if METHOD in ('prod', 'codeeraser') else 'forget'
TRAIN_ROOT = RUN / 'training'
existing = set(TRAIN_ROOT.glob('*')) if TRAIN_ROOT.exists() else set()
command = [sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node=1',
           '--master_port=29500', 'forget.py', '--config-path', './config', '--config-name', CONFIG,
           f'forget_loss={METHOD}', f'model_family={MODEL_FAMILY}', f'model_path={MODEL_PATH}',
           f'forget_data_path={FORGET}', f'retain_data_path={RETAIN}',
           f'LoRA.enabled={str(USE_LORA).lower()}', 'importance_file=null',
           f'tracking.output_dir={TRAIN_ROOT}', 'precision=auto']
# Thêm override tại đây nếu cần; chúng sẽ được lưu trong config của lần chạy.
# command += ['LoRA.r=320', 'LoRA.alpha=640', 'LoRA.targets=self_attn']
subprocess.run(command, check=True)
created = set(TRAIN_ROOT.glob('*')) - existing
assert len(created) == 1, f'Kiểm tra thư mục kết quả: {TRAIN_ROOT}'
TRAIN_RUN = created.pop()
TRAINED_MODEL = TRAIN_RUN / 'model'
(RUN / 'trained_model_path.txt').write_text(str(TRAINED_MODEL))
print('Checkpoint:', TRAINED_MODEL)
```

PROD mặc định: lr `5e-6`, 10 epochs, batch 1, accumulation 32, top-p `0.8`,
alpha `0`; CodeEraser: lr `3e-6`, tối đa 100 epochs theo config (không có early
stopping). Muốn chạy thử ngắn có thể thêm `num_epochs=1`, nhưng đó là override.
PROD giữ redistribution, frozen reference và loss gốc; phần QA chỉ tính trên
answer hợp lệ, bỏ prompt/padding và xử lý cả token answer đầu tiên.
`max_length` giới hạn chuỗi huấn luyện; `max_new_tokens=1024` là giới hạn sinh khi jailbreak.

## Cell 7 — jailbreak và judge sau unlearning

Dùng lại **toàn bộ dữ liệu gốc**, cùng seed/tham số generation và cùng judge.
Transformers/PEFT nạp adapter nếu checkpoint là LoRA; giữ nguyên model base ở
`MODEL_PATH` để adapter có thể được nạp lại.

```python
AFTER = RUN / 'jailbreak_after.jsonl'
LABELS_AFTER = RUN / 'labels_after.jsonl'
run_script('jailbreak.py', f'model_family={MODEL_FAMILY}', f'model_path={TRAINED_MODEL}',
           f'data_path={SOURCE}', f'output_path={AFTER}', f'torch_dtype={DTYPE}',
           f'num_gpus={JAILBREAK_GPUS}', f'batch_size={JAILBREAK_BATCH}',
           'max_new_tokens=1024', 'temperature=0.1', 'seed=42')
run_script('evals/judge_jsonl.py', '--input', AFTER, '--output', LABELS_AFTER,
           '--model', JUDGE_MODEL, '--url', JUDGE_URL)
```

## Cell 8 — xem kết quả và sao lưu

```python
import pandas as pd

comparison = []
for phase, path in [('before', LABELS_BEFORE), ('after', LABELS_AFTER)]:
    summary = json.loads(path.with_suffix('.summary.json').read_text())
    comparison.append(dict(phase=phase, total=summary['total'],
                           **summary['counts'], bad_rate=summary['bad_rate']))
comparison = pd.DataFrame(comparison)
comparison.to_csv(RUN / 'jailbreak_comparison.csv', index=False)
display(comparison)
run_script('summarize_run.py', TRAIN_RUN)
print('Tất cả kết quả:', RUN)
for path in (BEFORE, AFTER):
    print(path.name, json.loads(Path(str(path) + '.summary.json').read_text()))
print('Các bảng thống kê:', sorted(str(p.relative_to(RUN)) for p in RUN.rglob('*.csv')))

# Bản ZIP chứa bảng, config và kết quả generation; model nằm riêng trong RUN.
archive_path = RUN.parent / f'{RUN.name}_reports.zip'
with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in RUN.rglob('*'):
        if path.is_file() and 'model' not in path.relative_to(RUN).parts:
            archive.write(path, path.relative_to(RUN))
print('Báo cáo:', archive_path)
```

Chọn **Save Version → Save & Run All → Save output** trên Kaggle để giữ cả
checkpoint và kết quả trong `/kaggle/working`. Chỉ chạy cell chưa đảm bảo dữ liệu
còn sau khi session bị xóa. Tỷ lệ BAD trên tập đã dùng chọn forget là kết quả
trên tập đó; không coi là đánh giá held-out.

Trong `training/<run>/`: `model/`, config/source snapshot, thông tin tham số,
metrics từng sample trước/sau, log training, CPU/RAM/GPU và bảng tổng hợp.
MA/PPL trong tracking dùng teacher forcing (mặc định 512 token), không tương
đương ASR, HumanEval hay PDR của bài báo. Các benchmark bổ sung nằm trong
`bigcode-evaluation/`.

Nguồn method: [CodeEraser](https://arxiv.org/abs/2509.13755),
[PROD](https://ojs.aaai.org/index.php/AAAI/article/view/40398).
PROD dựa trên `jiangxxxue/PROD` commit `a581364`; giấy phép ở
`unlearning/licenses/PROD.txt`. Đã kiểm tra số học và Trainer trên CPU;
chưa kiểm chứng toàn bộ notebook bằng một phiên Kaggle GPU thực tế.
