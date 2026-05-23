from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


MALICIOUS_KEYWORDS = [
    "socket.connect",
    "socket.socket",
    "pty.spawn",
    "os.dup2",
    "requests.post",
    "requests.get",
    "subprocess",
    "os.system",
    "send(",
    "sendall(",
    "recv(",
    "bind(",
    "smtp",
    "ftp",
    "http://",
    "https://",
    "urllib",
    "urlopen",
    "listdir",
    "/etc/passwd",
    "/etc/shadow",
    "/.ssh",
    ".bash_history",
    "docker.sock",
    ".dockerenv",
    "id_rsa",
    "shutil.copy",
    "getpass",
    "gethostname",
    "environ",
    "os.getcwd",
    "os.getenv",
    "platform",
    "psutil",
    "winreg",
    "HKEY_",
    "ctypes",
    "exec(",
    "eval(",
    "os.popen",
    "os.execv",
    "os.fork",
    "base64",
    "marshal",
    "pickle",
    "zlib",
    "sql_connect",
    "sqlite3",
    "getCookie",
    "Cookie",
    "reverse",
    "shell",
    "backdoor",
    "passw",
    "secret",
    "login",
    "account",
    "paypal",
    "wallet",
    "AppData",
    "path2search",
    "key_words",
    "download",
    "persistent",
    "decode(",
    "encode(",
    "do_python3",
    "reverse_shell",
    "RemoteShell",
    "compress(",
    "cursor.execute",
]

LIBRARY_SIGNALS = [
    "def test_",
    "httpbin",
    "assert ",
    "pytest",
    "unittest",
    "AuthBase",
    "def __eq__",
    "def __ne__",
    "def __repr__",
    "atomic_open",
    "proxy_bypass",
    "PreparedRequest",
    "self.send(",
    "self.request(",
    "requests.Session",
]

BOILERPLATE_SIGNALS = [
    "CLASSIFIERS = [",
    "classifiers=['",
    "classifiers=[\"",
    "Development Status ::",
    "Intended Audience ::",
    "Programming Language :: Python",
]

QUESTION_TEMPLATE = """# Malware behavior: {groups}
{imports}

# Complete the following code:
{prompt_snippet}"""


@dataclass
class QaConversionConfig:
    cut_ratio: float = 0.3
    filter_groups: list[str] | None = None
    full_source_only: bool = False


@dataclass
class QaConversionReport:
    total: int = 0
    written: int = 0
    skipped: int = 0
    skip_reasons: Counter = field(default_factory=Counter)


class QaConverter:
    def __init__(self, config: QaConversionConfig | None = None) -> None:
        self.config = config or QaConversionConfig()

    def convert_record(self, record: dict[str, Any]) -> tuple[dict[str, str] | None, str | None]:
        imports = record.get("imports", "").strip()
        snippet = record.get("snippet", "").strip()
        groups = record.get("groups", [])

        if not snippet or not imports:
            return None, "missing_fields"

        snippet_lines = snippet.splitlines()
        total_lines = len(snippet_lines)
        cut = max(1, int(total_lines * self.config.cut_ratio))

        if total_lines - cut < 2:
            return None, "too_short"

        prompt_snippet = "\n".join(snippet_lines[:cut])
        completion = "\n".join(snippet_lines[cut:])
        combined = prompt_snippet + "\n" + completion

        if is_library_code(combined):
            return None, "library_code"
        if is_boilerplate(combined):
            return None, "boilerplate"
        if count_malicious(prompt_snippet) > count_malicious(completion):
            return None, "payload_in_question"

        question = QUESTION_TEMPLATE.format(
            imports=imports,
            groups=", ".join(groups),
            prompt_snippet=prompt_snippet,
        )
        return {"question": question, "answer": completion}, None

    def convert_jsonl(self, input_path: Path, output_path: Path) -> QaConversionReport:
        report = QaConversionReport()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        filter_groups = set(self.config.filter_groups or [])

        with input_path.open(encoding="utf-8") as input_file, output_path.open("w", encoding="utf-8") as output_file:
            for line in input_file:
                line = line.strip()
                if not line:
                    continue

                record = json.loads(line)
                report.total += 1

                if self.config.full_source_only and not record.get("full_source"):
                    report.skipped += 1
                    report.skip_reasons["missing_full_source"] += 1
                    continue

                if filter_groups and not filter_groups.intersection(record.get("groups", [])):
                    report.skipped += 1
                    report.skip_reasons["group_filtered"] += 1
                    continue

                converted, reason = self.convert_record(record)
                if converted is None:
                    report.skipped += 1
                    report.skip_reasons[reason or "unknown"] += 1
                    continue

                output_file.write(json.dumps(converted, ensure_ascii=False) + "\n")
                report.written += 1

        return report


def count_malicious(code: str) -> int:
    code_lower = code.lower()
    return sum(1 for keyword in MALICIOUS_KEYWORDS if keyword.lower() in code_lower)


def is_library_code(code: str) -> bool:
    return any(signal in code for signal in LIBRARY_SIGNALS)


def is_boilerplate(code: str) -> bool:
    return any(signal in code for signal in BOILERPLATE_SIGNALS)
