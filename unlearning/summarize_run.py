"""Rebuild plotting tables from durable rank files, including interrupted runs."""
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def read_jsonl(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # A hard kill can leave an incomplete final record.
                continue


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temp = path.with_suffix('.csv.tmp')
    with temp.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def numeric(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def describe(values):
    values = [v for v in values if v is not None]
    return dict(count=len(values), mean=statistics.mean(values) if values else None,
                std=statistics.stdev(values) if len(values) > 1 else None,
                min=min(values) if values else None, max=max(values) if values else None)


def summarize_run(directory):
    directory = Path(directory)
    rows = []
    for path in sorted(directory.glob('samples_*_rank*.csv')):
        with path.open(newline='', encoding='utf-8') as stream:
            rows.extend(row for row in csv.DictReader(stream)
                        if row.get('content_sha256') and row.get('metric_protocol')
                        and row.get('context_ma') is not None)
    rows.sort(key=lambda row: (row['phase'], row['split'], int(row['sample_index'])))
    write_csv(directory / 'samples.csv', rows)
    groups = defaultdict(list)
    for row in rows:
        groups[(row['phase'], row['split'])].append(row)
    summaries = []
    for (phase, split), group in groups.items():
        for part in ('answer', 'target', 'context'):
            for metric in ('loss', 'ppl', 'ma'):
                summary = dict(phase=phase, split=split, region=part, metric=metric,
                               rows=len(group), **describe([numeric(r.get(f'{part}_{metric}')) for r in group]))
                tokens = sum(numeric(r.get(f'{part}_tokens')) or 0 for r in group)
                nll = sum(numeric(r.get(f'{part}_nll_sum')) or 0 for r in group)
                correct = sum(numeric(r.get(f'{part}_correct')) or 0 for r in group)
                summary['total_tokens'] = tokens
                weighted_loss = nll / tokens if tokens else None
                summary['token_weighted_value'] = (
                    correct / tokens if metric == 'ma' and tokens else
                    weighted_loss if metric == 'loss' else
                    math.exp(weighted_loss) if metric == 'ppl' and weighted_loss is not None and weighted_loss < 709 else None)
                summaries.append(summary)
    write_csv(directory / 'sample_summary.csv', summaries)
    before = {(r['split'], r['sample_index'], r['content_sha256']): r for r in rows if r['phase'] == 'before'}
    deltas = []
    for after in (r for r in rows if r['phase'] == 'after'):
        key = (after['split'], after['sample_index'], after['content_sha256'])
        if key not in before:
            continue
        delta = dict(split=key[0], sample_index=key[1], content_sha256=key[2], sample_id=after['sample_id'])
        for part in ('answer', 'target', 'context'):
            for metric in ('loss', 'ppl', 'ma'):
                name = f'{part}_{metric}'
                old, new = numeric(before[key].get(name)), numeric(after.get(name))
                delta[f'{name}_before'], delta[f'{name}_after'] = old, new
                delta[f'{name}_delta'] = new - old if old is not None and new is not None else None
        deltas.append(delta)
    write_csv(directory / 'sample_deltas.csv', deltas)
    resources, gpus = [], []
    for path in sorted(directory.glob('resources_rank*.jsonl')):
        for row in read_jsonl(path):
            for gpu in row.pop('host_gpu_devices', []):
                gpus.append({key: row[key] for key in ('time_utc', 'elapsed_seconds', 'hostname', 'rank', 'phase', 'optimizer_step')} | gpu)
            resources.append(row)
    write_csv(directory / 'resources.csv', resources)
    write_csv(directory / 'resources_gpu.csv', gpus)
    gpu_groups = defaultdict(list)
    for row in gpus:
        gpu_groups[(row['hostname'], row['uuid'], row['phase'])].append(row)
    gpu_summaries = []
    for (hostname, uuid, phase), group in gpu_groups.items():
        for metric in ('utilization_percent', 'memory_used_mib', 'power_watts', 'temperature_c'):
            gpu_summaries.append(dict(hostname=hostname, gpu_uuid=uuid, phase=phase, metric=metric,
                                      **describe([numeric(r.get(metric)) for r in group])))
    write_csv(directory / 'resource_gpu_summary.csv', gpu_summaries)
    resource_groups = defaultdict(list)
    for row in resources:
        resource_groups[(row['rank'], row['phase'])].append(row)
    resource_summaries = []
    for (rank, phase), group in resource_groups.items():
        for metric in ('process_rss_bytes', 'process_cpu_percent', 'cuda_allocated_bytes', 'cuda_reserved_bytes'):
            resource_summaries.append(dict(rank=rank, phase=phase, metric=metric,
                                           **describe([numeric(r.get(metric)) for r in group])))
    write_csv(directory / 'resource_summary.csv', resource_summaries)
    logs = []
    for path in sorted(directory.glob('events_rank*.jsonl')):
        for event in read_jsonl(path):
            if event['event'] == 'trainer_log':
                values = event.pop('logs')
                logs.append({**event, **values})
    write_csv(directory / 'training_logs.csv', logs)
    statuses = [json.loads(p.read_text()) for p in directory.glob('status_rank*.json')]
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    expected = manifest.get('world_size', 1)
    complete = (len(statuses) == expected and all(s['status'] == 'completed' for s in statuses))
    training = [json.loads(p.read_text()) for p in directory.glob('training_summary_rank*.json')]
    durations = [r['training_wall_seconds_excluding_sample_evaluation'] for r in training]
    sizes_path = directory / 'dataset_sizes.json'
    sizes = json.loads(sizes_path.read_text()) if sizes_path.exists() else {}
    wall_seconds = max(durations) if durations else None
    training_summary = {
        'max_rank_training_wall_seconds_excluding_sample_evaluation': wall_seconds,
        'forget_dataset_size': sizes.get('forget'), 'retain_dataset_size': sizes.get('retain'),
        'amortized_seconds_per_forget_sample': wall_seconds / sizes['forget'] if wall_seconds is not None and sizes.get('forget') else None,
        'rank_summaries': training,
        'notes': ['Timing includes training loop, logging and checkpoint I/O; excludes added sample snapshots.',
                  'Amortized seconds/sample divides the whole run (all epochs) by unique forget rows; it is not individual latency.',
                  'Per-rank CUDA peaks must not be summed as if they were simultaneous.']}
    temporary = directory / 'training_summary.json.tmp'
    temporary.write_text(json.dumps(training_summary, indent=2), encoding='utf-8')
    temporary.replace(directory / 'training_summary.json')
    result = {'run_id': directory.name, 'complete': complete, 'expected_ranks': expected,
              'finished_ranks': len(statuses), 'sample_rows': len(rows), 'paired_samples': len(deltas),
              'resource_samples': len(resources),
              'notes': ['MA is teacher-forced argmax on text tokens; EL is evaluated separately.',
                        'GPU utilization/power are whole-device samples, not process attribution.',
                        'Resource averages are sample means, not time-weighted estimates.',
                        'A missing/failed rank means this is a partial run.']}
    temp = directory / 'summary.json.tmp'
    temp.write_text(json.dumps(result, indent=2), encoding='utf-8')
    temp.replace(directory / 'summary.json')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize_run(args.run_dir), indent=2))
