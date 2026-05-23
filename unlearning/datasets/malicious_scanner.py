from __future__ import annotations

import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .malicious_patterns import (
    BASE64_RE,
    GROUPS,
    HEX_ESCAPE_RE,
    LONG_STRING_RE,
    QUICK_KEYWORDS,
    SAFE_URL_RE,
)


@dataclass
class ScanReport:
    files_scanned: int = 0
    records_written: int = 0
    duplicates_skipped: int = 0
    group_counts: dict[str, int] = field(default_factory=lambda: {group: 0 for group in GROUPS})


class SnippetScanner:
    def __init__(self, extracted_dir: Path, workers: int = 4, max_file_size_kb: int = 512) -> None:
        self.extracted_dir = extracted_dir
        self.workers = workers
        self.max_file_size_bytes = max_file_size_kb * 1024

    def iter_python_files(self) -> list[Path]:
        return [
            path
            for path in sorted(self.extracted_dir.rglob("*.py"))
            if "__pycache__" not in path.parts and path.stat().st_size < self.max_file_size_bytes
        ]

    def scan_all(self) -> tuple[list[dict[str, Any]], ScanReport]:
        python_files = self.iter_python_files()
        report = ScanReport(files_scanned=len(python_files))
        records: list[dict[str, Any]] = []
        seen_hashes: set[str] = set()

        if not python_files:
            return records, report

        with ProcessPoolExecutor(max_workers=self.workers) as pool:
            futures = {
                pool.submit(process_file, str(path), str(self.extracted_dir)): path
                for path in python_files
            }

            for future in as_completed(futures):
                try:
                    file_records = future.result()
                except Exception:
                    file_records = []

                for record in file_records:
                    signature = hashlib.sha1(record["snippet"].encode("utf-8", "replace")).hexdigest()
                    if signature in seen_hashes:
                        report.duplicates_skipped += 1
                        continue
                    seen_hashes.add(signature)
                    records.append(record)
                    for group in record["groups"]:
                        report.group_counts[group] += 1

        report.records_written = len(records)
        records.sort(key=lambda record: (record.get("source_file", ""), record.get("line_start", 0)))
        return records, report

    def scan_to_jsonl(self, output_path: Path) -> ScanReport:
        records, report = self.scan_all()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(output_path, records)
        return report


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_file(py_path_str: str, base_str: str) -> list[dict[str, Any]]:
    py_path = Path(py_path_str)
    base = Path(base_str)

    try:
        source = py_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    if len(source) < 40:
        return []

    source_lower = source.lower()
    candidate_groups = [
        group
        for group, keywords in QUICK_KEYWORDS.items()
        if any(keyword in source_lower for keyword in keywords)
    ]
    if not candidate_groups:
        return []

    lines = source.splitlines()
    blocks_with_range = extract_blocks_with_range(source, lines)
    if not blocks_with_range:
        return []

    try:
        relative_path = str(py_path.relative_to(base))
    except ValueError:
        relative_path = str(py_path)

    parts = Path(relative_path).parts
    package = parts[0] if len(parts) > 0 else "unknown"
    version = parts[1] if len(parts) > 1 else "unknown"
    imports = extract_imports(lines)
    full_source = source.strip() if is_clean_file(source) else None

    records: list[dict[str, Any]] = []
    seen_in_file: set[str] = set()

    for block, start_line, end_line in blocks_with_range:
        stripped = block.strip()
        if not stripped:
            continue

        block_lower = stripped.lower()
        pre_groups = [
            group
            for group in candidate_groups
            if any(keyword in block_lower for keyword in QUICK_KEYWORDS[group])
        ]
        if not pre_groups:
            continue

        matched_groups = [
            group
            for group in pre_groups
            if any(pattern.search(stripped) for pattern in GROUPS[group])
        ]
        if not matched_groups:
            continue

        if not is_clean_snippet(stripped) or only_safe_urls(stripped):
            continue

        signature = hashlib.sha1(stripped.encode("utf-8", "replace")).hexdigest()
        if signature in seen_in_file:
            continue
        seen_in_file.add(signature)

        if len(stripped.splitlines()) >= 4:
            try:
                ast.parse(stripped)
            except SyntaxError:
                continue

        records.append(
            {
                "source_file": relative_path,
                "package": package,
                "version": version,
                "snippet_hash": signature,
                "groups": matched_groups,
                "line_start": start_line,
                "line_end": end_line,
                "total_lines": len(lines),
                "imports": imports,
                "context": get_surrounding_context(lines, start_line, end_line),
                "snippet": stripped,
                "full_source": full_source,
                "snippet_lines": len(stripped.splitlines()),
                "file_size_kb": round(py_path.stat().st_size / 1024, 2),
            }
        )

    return records


def is_clean_snippet(text: str) -> bool:
    if len(HEX_ESCAPE_RE.findall(text)) > 8:
        return False
    if BASE64_RE.search(text):
        return False
    if LONG_STRING_RE.search(text):
        return False

    lines = text.splitlines()
    real_lines = [line for line in lines if line.strip() and not line.strip().startswith("#")]
    return len(real_lines) >= 2 and len(lines) <= 60


def is_clean_file(text: str) -> bool:
    if len(HEX_ESCAPE_RE.findall(text)) > 30:
        return False
    if BASE64_RE.search(text):
        return False
    return len(text) <= 200 * 1024


def only_safe_urls(text: str) -> bool:
    urls = re.findall(r"https?://[^\s'\")\]]+", text)
    return bool(urls) and all(SAFE_URL_RE.match(url) for url in urls)


def extract_imports(lines: list[str]) -> str:
    imports = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")) and "import" in stripped:
            imports.append(line)
    return "\n".join(imports)


def get_surrounding_context(lines: list[str], start: int, end: int, window: int = 5) -> str:
    context_start = max(0, start - 1 - window)
    context_end = min(len(lines), end + window)
    numbered = []
    for line_number, line in enumerate(lines[context_start:context_end], context_start + 1):
        marker = ">>>" if start <= line_number <= end else "   "
        numbered.append(f"{marker} {line_number:4d} | {line}")
    return "\n".join(numbered)


def extract_blocks_with_range(source: str, lines: list[str]) -> list[tuple[str, int, int]]:
    results: list[tuple[str, int, int]] = []
    function_ranges: list[tuple[int, int]] = []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno
            end = getattr(node, "end_lineno", node.lineno + 1)
            results.append(("\n".join(lines[start - 1 : end]), start, end))
            function_ranges.append((start, end))

    covered = set()
    for start, end in function_ranges:
        covered.update(range(start, end + 1))

    buffer: list[str] = []
    buffer_start: int | None = None
    for line_number, line in enumerate(lines, 1):
        if line_number not in covered:
            if buffer_start is None:
                buffer_start = line_number
            buffer.append(line)
        else:
            if buffer and buffer_start is not None:
                chunk = "\n".join(buffer).strip()
                if chunk:
                    results.append((chunk, buffer_start, line_number - 1))
            buffer = []
            buffer_start = None

    if buffer and buffer_start is not None:
        chunk = "\n".join(buffer).strip()
        if chunk:
            results.append((chunk, buffer_start, len(lines)))

    return results
