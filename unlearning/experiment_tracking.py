"""Durable experiment artifacts and resource sampling (one writer per rank)."""
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from transformers import TrainerCallback

from codeeraser import encode_example


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(clean_json(data), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def append_jsonl(path, data):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(clean_json(data), ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sample_sha256(record):
    # Same ID across model runs; row index additionally distinguishes duplicates.
    text = json.dumps({'question': record['question'], 'answer': record['answer']},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def unique_run_dir(root, label):
    label = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(label))[:100]
    path = Path(root).resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
                                 + '_' + label + '_' + uuid.uuid4().hex[:12])
    path.mkdir(parents=True, exist_ok=False)
    return path


class CsvJournal:
    """Flush each row so completed samples survive a later exception/interrupt."""
    def __init__(self, path, fields):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('x', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.stream, fieldnames=fields)
        self.writer.writeheader()
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def write(self, row):
        self.writer.writerow(clean_json(row))
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self):
        self.stream.close()


class RunRecorder:
    def __init__(self, directory, rank=0, local_rank=0, interval=2.0):
        self.directory = Path(directory)
        self.rank = rank
        self.local_rank = local_rank
        self.interval = float(interval)
        if self.interval <= 0:
            raise ValueError('tracking.resource_interval_seconds must be positive.')
        self.phase = 'initialization'
        self.step = 0
        self.started = time.perf_counter()
        self.stop_event = threading.Event()
        self.resource_path = self.directory / f'resources_rank{rank}.jsonl'
        self.event_path = self.directory / f'events_rank{rank}.jsonl'
        self.errors = []
        self.process = psutil.Process()
        self.process.cpu_percent()
        self.device = local_rank if torch.cuda.is_available() else None
        self.thread = threading.Thread(target=self._monitor, daemon=True)

    def event(self, event, **values):
        append_jsonl(self.event_path, {'time_utc': utc_now(), 'elapsed_seconds': time.perf_counter() - self.started,
                                    'rank': self.rank, 'event': event, **values})

    def start(self):
        self.event('run_started')
        try:
            self.sample_resources()
        except Exception as exc:
            self.errors.append(f'{type(exc).__name__}: {exc}')
        self.thread.start()

    def sample_resources(self):
        row = {'time_utc': utc_now(), 'elapsed_seconds': time.perf_counter() - self.started,
               'rank': self.rank, 'hostname': platform.node(), 'phase': self.phase,
               'optimizer_step': self.step, 'process_rss_bytes': self.process.memory_info().rss,
               'process_cpu_percent': self.process.cpu_percent(),
               'host_ram_used_bytes': psutil.virtual_memory().used}
        if self.device is not None:
            row.update(cuda_device=self.device,
                       cuda_allocated_bytes=torch.cuda.memory_allocated(self.device),
                       cuda_reserved_bytes=torch.cuda.memory_reserved(self.device),
                       cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                       cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device))
        # Once per node. These readings describe whole devices, NOT this process.
        if self.local_rank == 0:
            try:
                query = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.used,power.draw,temperature.gpu',
                                        '--format=csv,noheader,nounits'], capture_output=True,
                                       text=True, timeout=2, check=True)
                devices = []
                for values in csv.reader(query.stdout.splitlines()):
                    def number(value):
                        try:
                            return float(value.strip())
                        except ValueError:
                            return None
                    devices.append(dict(index=values[0].strip(), uuid=values[1].strip(),
                                        utilization_percent=number(values[2]), memory_used_mib=number(values[3]),
                                        power_watts=number(values[4]), temperature_c=number(values[5])))
                row['host_gpu_devices'] = devices
            except (OSError, subprocess.SubprocessError) as exc:
                row['gpu_telemetry_unavailable'] = type(exc).__name__
        append_jsonl(self.resource_path, row)

    def _monitor(self):
        while not self.stop_event.wait(self.interval):
            try:
                self.sample_resources()
            except Exception as exc:
                self.errors.append(f'{type(exc).__name__}: {exc}')

    def finish(self, status, error=None):
        self.stop_event.set()
        self.thread.join(timeout=5)
        self.event('run_finished', status=status, error=error)
        atomic_json(self.directory / f'status_rank{self.rank}.json', {
            'status': status, 'finished_at': utc_now(), 'rank': self.rank,
            'total_wall_seconds': time.perf_counter() - self.started,
            'last_optimizer_step': self.step, 'error': error,
            'resource_monitor_errors': self.errors})


def environment_manifest():
    packages = {}
    for name in ('torch', 'transformers', 'accelerate', 'deepspeed', 'peft', 'datasets', 'psutil'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                                  text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain'], capture_output=True,
                               text=True, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision, dirty = None, None
    return {'created_at': utc_now(), 'python': platform.python_version(),
            'platform': platform.platform(), 'hostname': platform.node(), 'packages': packages,
            'cuda_version': torch.version.cuda, 'cuda_visible_devices': os.getenv('CUDA_VISIBLE_DEVICES'),
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            'git_commit': revision, 'git_dirty': bool(dirty), 'git_status': dirty}


def token_metrics(logits, labels, mask):
    labels = labels[1:]
    selected = mask[1:].bool() & (labels != -100)
    count = int(selected.sum())
    if count == 0:
        return {'tokens': 0, 'nll_sum': None, 'loss': None, 'ppl': None, 'correct': 0, 'ma': None}
    scores = logits[:-1][selected].float()
    truth = labels[selected]
    nll = F.cross_entropy(scores, truth, reduction='sum').item()
    correct = int((scores.argmax(-1) == truth).sum())
    loss = nll / count
    return {'tokens': count, 'nll_sum': nll, 'loss': loss,
            'ppl': math.exp(loss) if loss < 709 else None,
            'correct': correct, 'ma': correct / count}


class ExperimentCallback(TrainerCallback):
    def __init__(self, recorder, dataset, tokenizer, model_config, settings, world_size):
        self.recorder = recorder
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.settings = settings
        self.world_size = world_size
        self.training_started = None
        self.active_training_seconds = 0.0
        self.peak_allocated = 0
        self.peak_reserved = 0

    def _sync(self):
        if self.recorder.device is not None:
            torch.cuda.synchronize(self.recorder.device)

    def _pause_training(self):
        if self.training_started is not None:
            self._sync()
            if self.recorder.device is not None:
                self.peak_allocated = max(self.peak_allocated, torch.cuda.max_memory_allocated(self.recorder.device))
                self.peak_reserved = max(self.peak_reserved, torch.cuda.max_memory_reserved(self.recorder.device))
            self.active_training_seconds += time.perf_counter() - self.training_started
            self.training_started = None

    def _resume_training(self):
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        self._sync()
        if self.recorder.device is not None:
            torch.cuda.reset_peak_memory_stats(self.recorder.device)
        self.recorder.phase = 'training'
        self.training_started = time.perf_counter()

    @torch.no_grad()
    def snapshot(self, phase, state, model):
        self.recorder.phase = 'sample_evaluation'
        started = time.perf_counter()
        # Evaluation must not alter training RNG or dropout state.
        cpu_rng, python_rng, numpy_rng = torch.get_rng_state(), random.getstate(), np.random.get_state()
        cuda_rng = torch.cuda.get_rng_state(self.recorder.device) if self.recorder.device is not None else None
        was_training = model.training
        model.eval()
        fields = ['phase', 'optimizer_step', 'epoch', 'rank', 'split', 'sample_index', 'sample_id',
                  'content_sha256', 'metric_protocol', 'max_length', 'input_tokens', 'truncated', 'status']
        fields += [f'{part}_{key}' for part in ('answer', 'target', 'context')
                   for key in ('tokens', 'nll_sum', 'loss', 'ppl', 'correct', 'ma')]
        journal = CsvJournal(self.recorder.directory / f'samples_{phase}_rank{self.recorder.rank}.csv', fields)
        try:
            device = next(model.parameters()).device
            for split in ('forget', 'retain'):
                records = getattr(self.dataset, split, None)
                if records is None:
                    records = getattr(self.dataset, split + '_data', None)
                if records is None:
                    raise ValueError(f'Cannot find {split} records for per-sample metrics.')
                limit = self.settings.get('max_samples_per_split')
                size = len(records) if limit is None else min(len(records), int(limit))
                if size < 0:
                    raise ValueError('tracking.max_samples_per_split must be nonnegative or null.')
                indices = (list(range(len(records))) if limit is None else
                           sorted(random.Random(self.settings.get('sample_seed', 42)).sample(range(len(records)), size)))
                max_length = int(self.settings.get('sample_max_length', 512))
                for index in indices[self.recorder.rank::self.world_size]:
                    record = records[index]
                    base = encode_example(self.tokenizer, record['question'], record['answer'],
                                          self.model_config, max_length, require_target=False)
                    target = encode_example(self.tokenizer, record['question'], record['answer'],
                                            self.model_config, max_length, record.get('forget_spans'),
                                            require_target=False)
                    length = int(base['attention_mask'].sum())
                    prefix = (self.model_config['question_start_tag'] + record['question']
                              + self.model_config['question_end_tag'] + self.model_config['answer_tag'])
                    original_length = len(self.tokenizer(prefix + record['answer'])['input_ids'])
                    row = dict(phase=phase, optimizer_step=state.global_step, epoch=state.epoch,
                               rank=self.recorder.rank, split=split, sample_index=index,
                               sample_id=str(record.get('doc_id', record.get('id', index))),
                               content_sha256=sample_sha256(record), metric_protocol='teacher_forced_text_tokens_v1',
                               max_length=max_length, input_tokens=length, truncated=original_length > max_length,
                               status='ok' if base['secret_mask'][1:].any() else 'no_answer_tokens')
                    if length:
                        ids = base['input_ids'][:length].to(device)
                        logits = model(input_ids=ids.unsqueeze(0),
                                       attention_mask=torch.ones_like(ids).unsqueeze(0)).logits[0]
                        labels = base['labels'][:length].to(device)
                        masks = {'answer': base['secret_mask'], 'target': target['secret_mask'],
                                 'context': ~target['secret_mask'] & (base['labels'] != -100)}
                        for part, mask in masks.items():
                            row.update({f'{part}_{key}': value for key, value in
                                        token_metrics(logits, labels, mask[:length].to(device)).items()})
                        del logits
                    journal.write(row)
        finally:
            journal.close()
            model.train(was_training)
            torch.set_rng_state(cpu_rng)
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, self.recorder.device)
        self._sync()
        self.recorder.event('sample_evaluation_finished', phase=phase,
                            wall_seconds=time.perf_counter() - started, optimizer_step=state.global_step)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.settings.get('sample_metrics', True):
            self.snapshot('before', state, model)
        if self.recorder.device is not None:
            torch.cuda.reset_peak_memory_stats(self.recorder.device)
        self._resume_training()

    def on_step_end(self, args, state, control, **kwargs):
        self.recorder.step = state.global_step
        self.recorder.event('optimizer_step_finished', optimizer_step=state.global_step, epoch=state.epoch)

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.recorder.event('trainer_log', optimizer_step=state.global_step, epoch=state.epoch,
                            logs=logs or {})

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if self.settings.get('sample_metrics', True) and self.settings.get('sample_every_epoch', False):
            self._pause_training()
            self.snapshot(f'step{state.global_step}', state, model)
            self._resume_training()

    def on_train_end(self, args, state, control, model=None, **kwargs):
        self._pause_training()
        summary = {'training_wall_seconds_excluding_sample_evaluation': self.active_training_seconds,
                   'optimizer_steps': state.global_step, 'rank': self.recorder.rank,
                   'epoch': state.epoch}
        if self.recorder.device is not None:
            summary.update(cuda_training_peak_allocated_bytes=self.peak_allocated,
                           cuda_training_peak_reserved_bytes=self.peak_reserved)
        atomic_json(self.recorder.directory / f'training_summary_rank{self.recorder.rank}.json', summary)
        if self.settings.get('sample_metrics', True):
            self.snapshot('after', state, model)
        self.recorder.phase = 'saving'
