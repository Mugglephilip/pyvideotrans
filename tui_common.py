"""
tui_common.py -- pure helpers shared by the TUI app and its tests.

No heavy imports here on purpose: this module must stay importable without
PySide6 / torch / textual so the queue logic can be unit-tested cheaply.
"""

from __future__ import annotations

import glob
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional


# Status values used by the queue model.
QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"

STATUS_LABEL = {
    QUEUED: "排队中",
    RUNNING: "运行中",
    DONE: "已完成",
    FAILED: "失败",
    CANCELLED: "已取消",
}

TASK_LABEL = {"vtv": "视频翻译", "sts": "字幕翻译"}

# Pipeline stages in execution order for each task, with display labels.
STAGES_VTV = [
    ("prepare", "预处理"),
    ("recogn", "语音识别"),
    ("diariz", "说话人识别"),
    ("trans", "字幕翻译"),
    ("dubbing", "配音"),
    ("align", "音画对齐"),
    ("recogn2pass", "二次识别"),
    ("assembling", "合成"),
]
STAGES_STS = [
    ("prepare", "预处理"),
    ("trans", "字幕翻译"),
]

STAGE_LABEL = {k: v for k, v in STAGES_VTV + STAGES_STS}

_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def extract_percent(text: str) -> Optional[float]:
    """Return the first percentage found in a log line, if any."""
    if not text:
        return None
    m = _PERCENT_RE.search(text)
    if not m:
        return None
    try:
        return min(max(float(m.group(1)), 0.0), 100.0)
    except ValueError:
        return None


def stages_for(task: str) -> list[tuple[str, str]]:
    return list(STAGES_VTV if task == "vtv" else STAGES_STS)


def sanitize_basename(name: str) -> str:
    """Strip characters that cause problems in file/dir names."""
    return re.sub(r"[\s. #*?!:\"/]", "-", name)


def make_output_dir(noext_basename: str, root_dir: str, ts: Optional[str] = None) -> str:
    """Build a unique per-job output dir: <root>/output/<name>_<timestamp>."""
    ts = ts or time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(root_dir, "output", f"{sanitize_basename(noext_basename)}_{ts}")


def newest_file(directory: str) -> str:
    """Return the newest regular file inside *directory* ("" if none)."""
    files = [p for p in glob.glob(os.path.join(directory, "*")) if os.path.isfile(p)]
    if not files:
        return ""
    return max(files, key=os.path.getmtime)


@dataclass
class Job:
    """One item in the TUI queue."""

    job_id: int
    task: str  # "vtv" | "sts"
    name: str  # absolute input path
    source_lang: str
    target_lang: str
    voice_role: str = "No"
    status: str = QUEUED
    stage: str = ""
    percent: float = 0.0
    message: str = ""
    output: str = ""
    output_dir: str = ""
    overrides: dict = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    cancel_requested: bool = field(default=False, compare=False)
    proc: object = field(default=None, repr=False, compare=False)

    @property
    def display_name(self) -> str:
        return os.path.basename(self.name)

    @property
    def is_terminal(self) -> bool:
        return self.status in (DONE, FAILED, CANCELLED)

    def add_log(self, text: str, max_lines: int = 200) -> None:
        self.logs.append(text)
        if len(self.logs) > max_lines:
            del self.logs[: len(self.logs) - max_lines]


def default_overrides_from_params(params: dict) -> dict:
    """Pick the advanced-option defaults from videotrans/params.json."""
    def _int(key: str, default: int) -> int:
        try:
            return int(params.get(key, default))
        except (TypeError, ValueError):
            return default

    return {
        "recogn_type": _int("recogn_type", 0),
        "translate_type": _int("translate_type", 0),
        "tts_type": _int("tts_type", 0),
        "model_name": str(params.get("model_name") or "tiny"),
        "cuda": bool(params.get("cuda", False)),
    }
