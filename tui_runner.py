"""
tui_runner.py -- subprocess worker for the pyVideoTrans TUI.

The TUI keeps itself light (no PySide6 / torch imports) and delegates each
real vtv/sts run to this script.  It reuses the CLI's param builders, drives
the same task classes, and reports machine-readable JSON events on stdout:

    {"event":"stage","stage":"recogn","state":"start"}
    {"event":"log","text":"...","percent":45.0}
    {"event":"done","output":"/abs/result.mp4","output_dir":"/abs/out"}
    {"event":"error","message":"..."}
    {"event":"cancelled"}

Usage: python tui_runner.py /path/to/job.json
"""

import argparse
import json
import multiprocessing
import signal
import sys

from tui_common import extract_percent, newest_file

JOB_FILE = sys.argv[1] if len(sys.argv) > 1 else ""


def emit(event: dict) -> None:
    print(json.dumps(event, ensure_ascii=False), flush=True)


def load_job(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_args(job: dict) -> argparse.Namespace:
    """Build a CLI-style Namespace from the TUI job spec."""
    ov = job.get("overrides") or {}
    return argparse.Namespace(
        name=job["name"],
        output_dir=job.get("output_dir"),
        source_language_code=job.get("source_language_code"),
        target_language_code=job.get("target_language_code"),
        voice_role=job.get("voice_role"),
        # STT
        recogn_type=int(ov.get("recogn_type", 0)),
        detect_language="auto",
        model_name=str(ov.get("model_name", "tiny")),
        cuda=bool(ov.get("cuda", False)),
        remove_noise=False,
        enable_diariz=False,
        nums_diariz=-1,
        rephrase=0,
        fix_punc=False,
        # TTS
        tts_type=int(ov.get("tts_type", 0)),
        voice_rate="+0%",
        volume="+0%",
        pitch="+0Hz",
        voice_autorate=False,
        align_sub_audio=False,
        # Translation
        translate_type=int(ov.get("translate_type", 0)),
        # VTV extras
        video_autorate=False,
        is_separate=False,
        recogn2pass=False,
        subtitle_type=1,
        clear_cache=True,
    )


def build_params(job: dict, args: argparse.Namespace) -> dict:
    import cli as cli_mod

    common = cli_mod.build_common_params(args, output_dir=job.get("output_dir"))
    if job["task"] == "sts":
        return {**common, **cli_mod.build_sts_params(args)}
    return {**common, **cli_mod.build_vtv_params(args)}


def make_task(task: str, params: dict):
    if task == "sts":
        from videotrans.task.taskcfg import TaskCfgSTS
        from videotrans.task.translate_srt import TranslateSrt

        return TranslateSrt(cfg=TaskCfgSTS(**params), out_format=0)
    from videotrans.task.taskcfg import TaskCfgVTT
    from videotrans.task.trans_create import TransCreate

    return TransCreate(cfg=TaskCfgVTT(**params))


def run_pipeline(task: str, trk, stages: list) -> None:
    for stage, _label in stages:
        if app_cfg.exit_soft:
            return
        emit({"event": "stage", "stage": stage, "state": "start"})
        getattr(trk, stage)()
        if not app_cfg.exit_soft:
            emit({"event": "stage", "stage": stage, "state": "end"})


def main() -> int:
    global app_cfg

    if not JOB_FILE:
        print("usage: python tui_runner.py /path/to/job.json", file=sys.stderr)
        return 2

    from videotrans.configure import config
    from videotrans.configure.config import app_cfg as _app_cfg
    from videotrans.configure.base import BaseCon

    app_cfg = _app_cfg
    config.init_run()
    app_cfg.exec_mode = "tui"
    app_cfg.exit_soft = False
    app_cfg.current_status = "ing"

    # Route all task progress through our JSON protocol.
    _orig_signal = BaseCon.signal

    def _tui_signal(self, **kwargs):
        if app_cfg.exit_soft:
            return
        text = kwargs.get("text")
        if not text:
            return
        event = {"event": "log", "text": str(text)}
        percent = extract_percent(str(text))
        if percent is not None:
            event["percent"] = percent
        emit(event)

    BaseCon.signal = _tui_signal

    def _on_term(signum, frame):
        app_cfg.exit_soft = True
        emit({"event": "cancelling"})

    signal.signal(signal.SIGTERM, _on_term)

    job = load_job(JOB_FILE)
    task = job.get("task", "vtv")
    if task not in ("vtv", "sts"):
        emit({"event": "error", "message": f"unsupported task: {task}"})
        return 1

    try:
        stages = _STAGES[task]
        args = build_args(job)
        params = build_params(job, args)
        trk = make_task(task, params)
        run_pipeline(task, trk, stages)

        if app_cfg.exit_soft:
            emit({"event": "cancelled"})
            return 0

        trk.task_done()
        output = newest_file(params.get("target_dir") or "")
        emit(
            {
                "event": "done",
                "output": output,
                "output_dir": params.get("target_dir") or "",
            }
        )
        return 0
    except Exception as e:
        if app_cfg.exit_soft:
            emit({"event": "cancelled"})
            return 0
        emit({"event": "error", "message": str(e)})
        return 1


_STAGES = {
    "vtv": ["prepare", "recogn", "diariz", "trans", "dubbing", "align", "recogn2pass", "assembling"],
    "sts": ["prepare", "trans"],
}


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    sys.exit(main())
