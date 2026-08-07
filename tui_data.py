"""
tui_data.py -- lightweight data providers for the TUI.

Deliberately avoids importing the videotrans package (that would pull in
PySide6 / torch).  Languages and channel names come from a one-shot
`cli.py --list ...` subprocess; voice roles come straight from the
`videotrans/voicejson/*.json` data files.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parent
VOICE_DIR = ROOT_DIR / "videotrans" / "voicejson"
PARAMS_FILE = ROOT_DIR / "videotrans" / "params.json"

# tts_type -> voice json file + whether it is structured per-language.
VOICE_FILES: dict[int, tuple[str, bool]] = {
    0: ("edge_tts.json", True),      # Edge-TTS
    7: ("piper.json", True),         # Piper
    9: ("supertonic.json", False),   # Supertonic
    11: ("minimaxi.json", True),     # MiniMax
    15: ("doubao2.json", True),      # Doubao 2.0
    16: ("qwen3tts.json", False),    # Qwen TTS
}

_LANG_CODE_RE = re.compile(r"^[a-zA-Z]{2,3}(?:-[a-zA-Z0-9]{2,8})*$")


def read_params() -> dict:
    try:
        return json.loads(PARAMS_FILE.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}


def _run_cli_list(kind: str) -> list[str]:
    cmd = [sys.executable, str(ROOT_DIR / "cli.py"), "--list", kind]
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        timeout=120,
    )
    return (proc.stdout or "").splitlines()


def list_languages() -> list[tuple[str, str]]:
    """Return [(code, name), ...] for the language picker."""
    fallback = [
        ("en", "English"),
        ("zh-cn", "简体中文"),
        ("zh-tw", "繁體中文"),
        ("ja", "日本語"),
        ("ko", "한국어"),
        ("fr", "Français"),
        ("de", "Deutsch"),
        ("ru", "Русский"),
        ("es", "Español"),
    ]
    try:
        lines = _run_cli_list("languages")
    except Exception:
        return fallback
    result = []
    for line in lines:
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and _LANG_CODE_RE.match(parts[0]):
            result.append((parts[0], parts[1].strip()))
    return result or fallback


def list_providers() -> dict[str, list[str]]:
    """Return {recognition: [...], translation: [...], tts: [...]}."""
    result: dict[str, list[str]] = {"recognition": [], "translation": [], "tts": []}
    try:
        lines = _run_cli_list("providers")
    except Exception:
        return result
    section = None
    for line in lines:
        low = line.lower()
        if "(stt)" in low or "语音识别" in line:
            section = "recognition"
            continue
        if "(translation)" in low or "翻译" in line and "(tts)" not in low:
            section = "translation"
            continue
        if "(tts)" in low or "配音" in line:
            section = "tts"
            continue
        parts = line.strip().split(" = ", 1)
        if section and len(parts) == 2:
            result[section].append(parts[1].strip())
    return result


def voice_roles(tts_type: int, target_lang: str) -> list[tuple[str, str]]:
    """Return [(label, role_code), ...] for the voice picker."""
    filename, per_lang = VOICE_FILES.get(tts_type, VOICE_FILES[0])
    path = VOICE_DIR / filename
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return [("No", "No")]
    if not isinstance(data, dict) or not data:
        return [("No", "No")]
    if per_lang:
        lang = (target_lang or "en").split("-")[0]
        roles = data.get(lang) if isinstance(data.get(lang), dict) else {}
    else:
        roles = data
    merged = {"No": "No"}
    for label, code in roles.items():
        merged[str(label)] = str(code)
    return list(merged.items())


def provider_name(providers: dict[str, list[str]], kind: str, index: int) -> str:
    names = providers.get(kind, [])
    return names[index] if 0 <= index < len(names) else f"#{index}"


def ensure_output_dir(base: str, noext_basename: str) -> str:
    from tui_common import make_output_dir

    out = make_output_dir(noext_basename, base)
    os.makedirs(out, exist_ok=True)
    return out
