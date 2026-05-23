from __future__ import annotations

import argparse
from dataclasses import asdict
import logging
from pathlib import Path

from .malicious_archive import ArchiveExtractor
from .malicious_filter import SnippetFilter, SnippetFilterConfig
from .malicious_qa import QaConversionConfig, QaConverter
from .malicious_scanner import SnippetScanner


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_REPO_DIR = DEFAULT_DATA_DIR / "pypi_malregistry"
DEFAULT_EXTRACTED_DIR = DEFAULT_DATA_DIR / "pypi_extracted"
DEFAULT_RAW_JSONL = DEFAULT_DATA_DIR / "malicious_snippets.jsonl"
DEFAULT_CLEAN_JSONL = DEFAULT_DATA_DIR / "malicious_clean.jsonl"
DEFAULT_QA_JSONL = DEFAULT_DATA_DIR / "malicious_qa.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a malicious-code JSONL dataset from PyPI malware package archives."
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract", help="Extract package archives safely.")
    extract_parser.add_argument("--repo-dir", type=Path, default=DEFAULT_REPO_DIR)
    extract_parser.add_argument("--out-dir", type=Path, default=DEFAULT_EXTRACTED_DIR)

    scan_parser = subparsers.add_parser("scan", help="Scan extracted Python files for malicious snippets.")
    scan_parser.add_argument("--extracted-dir", type=Path, default=DEFAULT_EXTRACTED_DIR)
    scan_parser.add_argument("--output", type=Path, default=DEFAULT_RAW_JSONL)
    scan_parser.add_argument("--workers", type=int, default=4)
    scan_parser.add_argument("--max-file-size-kb", type=int, default=512)

    filter_parser = subparsers.add_parser("filter", help="Filter noisy snippets and write clean JSONL.")
    filter_parser.add_argument("--input", type=Path, default=DEFAULT_RAW_JSONL)
    filter_parser.add_argument("--output", type=Path, default=DEFAULT_CLEAN_JSONL)
    add_filter_arguments(filter_parser)

    convert_parser = subparsers.add_parser("convert", help="Convert clean snippets to question/answer JSONL.")
    convert_parser.add_argument("--input", type=Path, default=DEFAULT_CLEAN_JSONL)
    convert_parser.add_argument("--output", type=Path, default=DEFAULT_QA_JSONL)
    convert_parser.add_argument("--cut-ratio", type=float, default=0.3)
    convert_parser.add_argument("--filter-groups", nargs="*", default=None)
    convert_parser.add_argument("--full-source-only", action="store_true")

    all_parser = subparsers.add_parser("all", help="Run extract, scan, filter, and convert.")
    all_parser.add_argument("--repo-dir", type=Path, default=DEFAULT_REPO_DIR)
    all_parser.add_argument("--extracted-dir", type=Path, default=DEFAULT_EXTRACTED_DIR)
    all_parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW_JSONL)
    all_parser.add_argument("--clean-output", type=Path, default=DEFAULT_CLEAN_JSONL)
    all_parser.add_argument("--qa-output", type=Path, default=DEFAULT_QA_JSONL)
    all_parser.add_argument("--workers", type=int, default=4)
    all_parser.add_argument("--max-file-size-kb", type=int, default=512)
    all_parser.add_argument("--cut-ratio", type=float, default=0.3)
    all_parser.add_argument("--filter-groups", nargs="*", default=None)
    all_parser.add_argument("--full-source-only", action="store_true")
    add_filter_arguments(all_parser)

    return parser


def add_filter_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--min-snippet-lines", type=int, default=3)
    parser.add_argument("--max-snippet-lines", type=int, default=60)
    parser.add_argument("--max-snippet-chars", type=int, default=3000)
    parser.add_argument("--max-context-lines", type=int, default=50)
    parser.add_argument("--max-full-source-lines", type=int, default=60)
    parser.add_argument("--deduplicate-by-snippet", action="store_true")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    if args.command == "extract":
        run_extract(args.repo_dir, args.out_dir)
    elif args.command == "scan":
        run_scan(args.extracted_dir, args.output, args.workers, args.max_file_size_kb)
    elif args.command == "filter":
        run_filter(args.input, args.output, args)
    elif args.command == "convert":
        run_convert(args.input, args.output, args.cut_ratio, args.filter_groups, args.full_source_only)
    elif args.command == "all":
        run_extract(args.repo_dir, args.extracted_dir)
        run_scan(args.extracted_dir, args.raw_output, args.workers, args.max_file_size_kb)
        run_filter(args.raw_output, args.clean_output, args)
        run_convert(args.clean_output, args.qa_output, args.cut_ratio, args.filter_groups, args.full_source_only)


def run_extract(repo_dir: Path, out_dir: Path) -> None:
    report = ArchiveExtractor(repo_dir=repo_dir, output_dir=out_dir).extract_all()
    LOGGER.info("Archive extraction report: %s", asdict(report))


def run_scan(extracted_dir: Path, output_path: Path, workers: int, max_file_size_kb: int) -> None:
    scanner = SnippetScanner(
        extracted_dir=extracted_dir,
        workers=workers,
        max_file_size_kb=max_file_size_kb,
    )
    report = scanner.scan_to_jsonl(output_path)
    LOGGER.info("Snippet scan report: %s", asdict(report))


def run_filter(input_path: Path, output_path: Path, args: argparse.Namespace) -> None:
    config = SnippetFilterConfig(
        min_snippet_lines=args.min_snippet_lines,
        max_snippet_lines=args.max_snippet_lines,
        max_snippet_chars=args.max_snippet_chars,
        max_context_lines=args.max_context_lines,
        max_full_source_lines=args.max_full_source_lines,
        deduplicate_by_full_source=not args.deduplicate_by_snippet,
    )
    report = SnippetFilter(config).filter_jsonl(input_path, output_path)
    LOGGER.info("Snippet filter report: %s", asdict(report))


def run_convert(
    input_path: Path,
    output_path: Path,
    cut_ratio: float,
    filter_groups: list[str] | None,
    full_source_only: bool,
) -> None:
    config = QaConversionConfig(
        cut_ratio=cut_ratio,
        filter_groups=filter_groups,
        full_source_only=full_source_only,
    )
    report = QaConverter(config).convert_jsonl(input_path, output_path)
    LOGGER.info("QA conversion report: %s", asdict(report))


if __name__ == "__main__":
    main()
