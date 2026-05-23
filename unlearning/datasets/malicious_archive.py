from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tarfile
import zipfile


SUPPORTED_ARCHIVE_SUFFIXES = (
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".tbz2",
    ".tar.xz",
    ".txz",
    ".zip",
    ".whl",
    ".egg",
)

TAR_SUFFIXES = (".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")


@dataclass
class ArchiveExtractionReport:
    total: int = 0
    extracted: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)


class ArchiveExtractor:
    def __init__(self, repo_dir: Path, output_dir: Path) -> None:
        self.repo_dir = repo_dir
        self.output_dir = output_dir

    @staticmethod
    def is_supported(path: Path) -> bool:
        name = path.name.lower()
        return any(name.endswith(suffix) for suffix in SUPPORTED_ARCHIVE_SUFFIXES)

    @staticmethod
    def _safe_extract_tar(source: Path, destination: Path) -> None:
        with tarfile.open(source) as archive:
            members = []
            for member in archive.getmembers():
                member_path = Path(member.name)
                if member_path.is_absolute() or ".." in member_path.parts:
                    continue
                members.append(member)
            archive.extractall(path=destination, members=members)

    @staticmethod
    def _safe_extract_zip(source: Path, destination: Path) -> None:
        with zipfile.ZipFile(source) as archive:
            for member in archive.infolist():
                member_path = Path(member.filename)
                if member_path.is_absolute() or ".." in member_path.parts:
                    continue
                archive.extract(member, destination)

    @classmethod
    def extract_one(cls, source: Path, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        name = source.name.lower()
        if any(name.endswith(suffix) for suffix in TAR_SUFFIXES):
            cls._safe_extract_tar(source, destination)
        else:
            cls._safe_extract_zip(source, destination)

    def iter_archives(self) -> list[Path]:
        return [
            path
            for path in sorted(self.repo_dir.rglob("*"))
            if path.is_file() and self.is_supported(path)
        ]

    def extract_all(self) -> ArchiveExtractionReport:
        report = ArchiveExtractionReport()
        archives = self.iter_archives()
        report.total = len(archives)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for source in archives:
            relative_path = source.relative_to(self.repo_dir)
            destination = self.output_dir / relative_path.parent
            sentinel = destination / f".extracted_{source.name}"

            if sentinel.exists():
                report.skipped += 1
                continue

            try:
                self.extract_one(source, destination)
                sentinel.touch()
                report.extracted += 1
            except Exception as error:
                report.failed += 1
                report.errors.append((str(relative_path), str(error)))

        return report
