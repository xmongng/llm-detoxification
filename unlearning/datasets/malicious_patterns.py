from __future__ import annotations

import re


GROUPS_RAW = {
    "Command Execution": [
        r"\bos\.system\s*\(",
        r"\bsubprocess\.(run|call|Popen|check_output|check_call)\s*\(",
        r"\bos\.popen\s*\(",
        r"\bcommands\.(getoutput|getstatusoutput)\s*\(",
        r"\bpexpect\.spawn\s*\(",
        r"shell\s*=\s*True",
        r"\bpty\.spawn\s*\(",
        r"\bos\.execv[pe]?\s*\(",
        r"cmd\s*/[cCkK]\s+(?:del|rd|rmdir|erase|format|reg)",
        r"\bpowershell(?:\.exe)?\s+-(?:enc|exec|nop|w\s+hidden)",
        r"\bwmic\b",
        r"\bcertutil\s+-(?:decode|urlcache)",
        r"\bbitsadmin\s+/transfer",
        r"\bmshta\.exe\b",
        r"\bregsvr32\b",
        r"\bwscript\b|\bcscript\b",
    ],
    "Information Stealing": [
        r"\bplatform\.(node|uname|machine|processor|version)\s*\(",
        r"\bgetpass\.(getuser|getpass)\s*\(",
        r"\bpwd\.getpwuid\s*\(",
        r"os\.environ(?:\[|\s*\.get)\s*\(",
        r"[\"'](?:HOME|USERNAME|USERPROFILE|APPDATA|COMPUTERNAME)[\"']",
        r"\.ssh[/\\](?:id_rsa|id_ed25519|known_hosts|config)",
        r"\.aws[/\\]credentials",
        r"\.git-credentials",
        r"\bkeyring\.(get_password|set_password)",
        r"\bpyperclip\.(copy|paste)\s*\(",
        r"\bwin32clipboard\.",
        r"(?:DISCORD|SLACK|GITHUB|STRIPE|TWILIO)_TOKEN",
        r"(?:password|passwd|api[_\-]?key|secret[_\-]?key)\s*=\s*[\"'][^\"']{4,}",
        r"(?:\.bash_history|\.zsh_history|\.psql_history)",
        r"\bsys\.platform\b",
        r"\bplatform\.system\s*\(\s*\)",
        r"\bsys\.version\b|\bsys\.version_info\b",
        r"\bleveldb\b|\bplyvel\b",
        r"Local\s+State|Login\s+Data|Cookies|Web\s+Data",
        r"roblox.*\.json|roblosecurity",
        r"\.minecraft[/\\](?:launcher_accounts|authlib)",
        r"exodus.*wallet|electrum.*wallet|metamask",
        r"AppData[/\\]Roaming[/\\](?:discord|telegram|signal)",
        r"keylogger|GetAsyncKeyState|SetWindowsHookEx",
        r"\bpynput\.keyboard\b",
        r"mss\.mss\s*\(\s*\)|ImageGrab\.grab\s*\(",
        r"wifi.*password|netsh\s+wlan\s+show",
    ],
    "File Operation": [
        r"\bopen\s*\([^)]{0,120}[\"'][war][b+]?[\"']",
        r"\bshutil\.(copy2?|move|rmtree|copytree|disk_usage)\s*\(",
        r"\bos\.(remove|unlink|rename|makedirs|mkdir|walk)\s*\(",
        r"Path\([^)]{0,80}\)\.(write_text|write_bytes|unlink|rmdir|mkdir)\s*\(",
        r"\bglob\.glob\s*\(",
        r"\btempfile\.(mkstemp|NamedTemporaryFile|mkdtemp)\s*\(",
        r"\bzipfile\.ZipFile\s*\([^)]*[\"']w",
        r"\btarfile\.open\s*\([^)]*[\"']w",
        r"\bfcntl\.(flock|lockf)\s*\(",
        r"\bmsvcrt\.locking\s*\(",
        r"\bwatchdog\.observers\b|\bwatchdog\.events\b",
        r"\binotify\b|\bpyinotify\b",
        r"\bwin32file\.(ReadDirectoryChangesW|FindFirstFile)",
        r"os\.stat\s*\(|os\.path\.getsize\s*\(",
        r"winreg\.(OpenKey|SetValueEx|CreateKey)",
        r"_winreg\.",
    ],
    "Infiltration": [
        r"install_requires\s*=\s*\[",
        r"setup_requires\s*=\s*\[",
        r"extras_require\s*=\s*\{",
        r"\bpip\._internal\b",
        r"site-packages[/\\](?![\w]+\.dist-info)",
        r"\.pth\s*[\"'].*write|open.*\.pth",
        r"usercustomize|sitecustomize",
        r"PYTHONSTARTUP|PYTHONPATH\s*=",
        r"\.egg-link\b",
        r"pkg_resources\.working_set",
        r"importlib\.metadata.*files\(",
        r"__path__\s*=\s*\[|__file__\s*=\s*[\"']",
    ],
}

QUICK_KEYWORDS = {
    "Command Execution": [
        "os.system",
        "subprocess",
        "os.popen",
        "pty.spawn",
        "os.execv",
        "shell=True",
        "cmd /c",
        "cmd /k",
        "powershell",
        "wmic",
        "certutil",
        "bitsadmin",
        "mshta",
        "regsvr32",
        "wscript",
        "cscript",
    ],
    "Information Stealing": [
        "environ",
        "getuser",
        "getpass",
        ".ssh/",
        "credentials",
        "api_key",
        "secret_key",
        "password",
        "keyring",
        "clipboard",
        "sys.platform",
        "platform.system",
        "leveldb",
        "roblosecurity",
        "roblox",
        "login data",
        "local state",
        "metamask",
        "exodus",
        "electrum",
        "discord",
        "telegram",
        "pynput",
        "imagegrab",
        "mss.mss",
        "wifi",
    ],
    "File Operation": [
        "open(",
        "shutil.",
        "os.remove",
        "os.unlink",
        "os.walk",
        "write_bytes",
        "write_text",
        "zipfile",
        "tarfile",
        "glob",
        "watchdog",
        "inotify",
        "win32file",
        "winreg",
        "_winreg",
    ],
    "Infiltration": [
        "install_requires",
        "setup_requires",
        "extras_require",
        "pip._internal",
        "site-packages",
        "sitecustomize",
        "usercustomize",
        "pythonstartup",
        "pythonpath",
        ".pth",
        "egg-link",
        "pkg_resources",
        "__path__",
        "importlib.metadata",
    ],
}

GROUPS = {
    group: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for group, patterns in GROUPS_RAW.items()
}

HEX_ESCAPE_RE = re.compile(r"\\x[0-9a-fA-F]{2}")
BASE64_RE = re.compile(r"(?<![\"'\w])[A-Za-z0-9+/]{64,}={0,2}(?![\"'\w])")
LONG_STRING_RE = re.compile(r"[\"'][^\"']{300,}[\"']")
SAFE_URL_RE = re.compile(r"https?://(?:pypi\.org|python\.org|docs\.python\.org|pkg-resources)")

BASE64_LONG_RE = re.compile(r"[A-Za-z0-9+/]{80,}={0,2}")
HEX_MANY_RE = re.compile(r"(\\x[0-9a-fA-F]{2}){4,}")
HEX_RAW_RE = re.compile(r"(?:0x[0-9a-fA-F]{2},?\s*){6,}")
NOISE_LINE_RE = re.compile(r"^\s*(?:#.*|b'[A-Za-z0-9+/=]{50,}'|\"[A-Za-z0-9+/=]{50,}\")\s*$")
OBFUSCATED_VAR_RE = re.compile(r"\b[O0]{6,}\b")
INTEGER_PAYLOAD_RE = re.compile(r"(?:\d{4,}\s+){10,}")
CHR_CHAIN_RE = re.compile(r"(?:chr\(\d+\)[,\s+]*){5,}")
EVAL_OBFUSCATION_RE = re.compile(r"(?:eval|exec)\s*\(\s*[\"']?\s*\\?[\"']?\s*\.join")

BEHAVIOR_SIGNALS = [
    "exec(",
    "eval(",
    "__import__",
    "subprocess",
    "os.system",
    "socket",
    "requests.",
    "urllib",
    "b64decode",
    "thread(",
    "popen",
    "send(",
    "connect(",
    "environ",
    "getuser",
    ".ssh",
    "credentials",
    "token",
    "steal",
    "exfil",
    "payload",
    "reverse",
    "shell",
    "keylog",
    "screenshot",
]


def count_real_lines(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip() and not line.strip().startswith("#"))


def is_noisy_text(text: str) -> bool:
    if BASE64_LONG_RE.search(text):
        return True
    if HEX_MANY_RE.search(text):
        return True
    if HEX_RAW_RE.search(text):
        return True
    if LONG_STRING_RE.search(text):
        return True
    if OBFUSCATED_VAR_RE.search(text):
        return True
    if INTEGER_PAYLOAD_RE.search(text):
        return True
    if CHR_CHAIN_RE.search(text):
        return True
    if EVAL_OBFUSCATION_RE.search(text):
        return True

    lines = text.splitlines()
    if not lines:
        return False
    noise_lines = sum(1 for line in lines if NOISE_LINE_RE.match(line))
    return noise_lines / len(lines) > 0.4


def has_actual_malicious_logic(snippet: str) -> bool:
    snippet_lower = snippet.lower()
    return any(signal in snippet_lower for signal in BEHAVIOR_SIGNALS)
