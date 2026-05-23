from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

from .malicious_patterns import count_real_lines, has_actual_malicious_logic, is_noisy_text, BASE64_LONG_RE


@dataclass
class SnippetFilterConfig:
    min_snippet_lines: int = 3
    max_snippet_lines: int = 60
    max_snippet_chars: int = 3000
    max_context_lines: int = 50
    max_full_source_lines: int = 60
    deduplicate_by_full_source: bool = True


@dataclass
class SnippetFilterReport:
    total: int = 0
    kept: int = 0
    dropped_noisy: int = 0
    dropped_short: int = 0
    dropped_long: int = 0
    dropped_false_positive: int = 0
    dropped_duplicate: int = 0
    invalid_json: int = 0
    skip_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def dropped_total(self) -> int:
        return (
            self.dropped_noisy
            + self.dropped_short
            + self.dropped_long
            + self.dropped_false_positive
            + self.dropped_duplicate
        )


class SnippetFilter:
    def __init__(self, config: SnippetFilterConfig | None = None) -> None:
        self.config = config or SnippetFilterConfig()
        self._seen_hashes: set[str] = set()

    def filter_record(self, record: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        snippet = record.get("snippet", "")
        groups = record.get("groups", [])

        if is_noisy_text(snippet):
            return None, "noisy"
        if self.is_too_short(snippet):
            return None, "short"
        if self.is_too_long(snippet):
            return None, "long"
        if self.is_setup_false_positive(snippet, groups):
            return None, "false_positive"

        signature = self.make_dedup_key(record)
        if signature in self._seen_hashes:
            return None, "duplicate"
        self._seen_hashes.add(signature)

        clean_record = {
            "groups": groups,
            "snippet": snippet,
            "imports": record.get("imports", ""),
            "context": self.clean_context(record.get("context", "")),
            "snippet_lines": record.get("snippet_lines", count_real_lines(snippet)),
        }

        full_source = record.get("full_source") or ""
        if (
            full_source
            and len(full_source.splitlines()) <= self.config.max_full_source_lines
            and not is_noisy_text(full_source)
        ):
            clean_record["full_source"] = full_source

        return clean_record, None

    def filter_jsonl(self, input_path: Path, output_path: Path) -> SnippetFilterReport:
        report = SnippetFilterReport()
        self._seen_hashes.clear()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with input_path.open(encoding="utf-8") as input_file, output_path.open("w", encoding="utf-8") as output_file:
            for line in input_file:
                line = line.strip()
                if not line:
                    continue
                report.total += 1

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    report.invalid_json += 1
                    continue

                filtered_record, reason = self.filter_record(record)
                if filtered_record is None:
                    self._record_drop(report, reason or "unknown")
                    continue

                output_file.write(json.dumps(filtered_record, ensure_ascii=False) + "\n")
                report.kept += 1

        return report

    def clean_context(self, context: str) -> str:
        if not context:
            return ""

        cleaned = []
        for line in context.splitlines():
            if BASE64_LONG_RE.search(line):
                cleaned.append(line[:20].rstrip() + " ... [encoded payload truncated]")
            else:
                cleaned.append(line)

        return "\n".join(cleaned[: self.config.max_context_lines])

    def is_too_short(self, snippet: str) -> bool:
        return count_real_lines(snippet) < self.config.min_snippet_lines

    def is_too_long(self, snippet: str) -> bool:
        return len(snippet) > self.config.max_snippet_chars or count_real_lines(snippet) > self.config.max_snippet_lines

    @staticmethod
    def is_setup_false_positive(snippet: str, groups: list[str]) -> bool:
        return groups == ["Infiltration"] and not has_actual_malicious_logic(snippet)

    def make_dedup_key(self, record: dict[str, Any]) -> str:
        if self.config.deduplicate_by_full_source:
            value = (record.get("full_source") or record.get("snippet", "")).strip()
        else:
            value = record.get("snippet", "").strip()
        return hashlib.sha1(value.encode("utf-8", "replace")).hexdigest()

    @staticmethod
    def _record_drop(report: SnippetFilterReport, reason: str) -> None:
        report.skip_reasons[reason] = report.skip_reasons.get(reason, 0) + 1
        if reason == "noisy":
            report.dropped_noisy += 1
        elif reason == "short":
            report.dropped_short += 1
        elif reason == "long":
            report.dropped_long += 1
        elif reason == "false_positive":
            report.dropped_false_positive += 1
        elif reason == "duplicate":
            report.dropped_duplicate += 1
