"""
End-to-end queue lifecycle tests with a stubbed runner script.

The real `tui_runner.py` (and therefore videotrans/PySide6) is replaced by a
tiny script that emits the same JSON protocol, so these tests exercise the
dispatcher, progress handling and cancellation without heavy dependencies.
"""

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("textual")

import tui_app
from tui_app import PyVideoTransTUI
from tui_common import CANCELLED, DONE, RUNNING


FAKE_RUNNER_QUICK = """
import json, sys
spec = json.load(open(sys.argv[1]))
print(json.dumps({"event": "stage", "stage": "prepare", "state": "start"}), flush=True)
print(json.dumps({"event": "log", "text": "处理中 50%", "percent": 50.0}), flush=True)
out = spec["output_dir"]
open(out + "/result.mp4", "w").close()
print(json.dumps({"event": "done", "output": out + "/result.mp4", "output_dir": out}), flush=True)
"""

FAKE_RUNNER_SLOW = """
import json, sys, time
spec = json.load(open(sys.argv[1]))
print(json.dumps({"event": "stage", "stage": "dubbing", "state": "start"}), flush=True)
print(json.dumps({"event": "log", "text": "配音中 10%", "percent": 10.0}), flush=True)
time.sleep(8)
print(json.dumps({"event": "done", "output": "", "output_dir": spec["output_dir"]}), flush=True)
"""


def _make_job_spec(tmp_path: Path) -> dict:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    return {
        "task": "vtv",
        "name": str(video),
        "source_language_code": "zh-cn",
        "target_language_code": "en",
        "voice_role": "No",
        "overrides": {"recogn_type": 0, "translate_type": 0, "tts_type": 0, "model_name": "tiny", "cuda": False},
    }


def _install_runner(tmp_path: Path, script: str, monkeypatch) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    runner = repo / "tui_runner.py"
    runner.write_text(script, encoding="utf-8")
    monkeypatch.setattr(tui_app, "ROOT_DIR", repo)
    return repo


def test_job_runs_to_done(tmp_path, monkeypatch):
    _install_runner(tmp_path, FAKE_RUNNER_QUICK, monkeypatch)
    spec = _make_job_spec(tmp_path)

    async def _run():
        app = PyVideoTransTUI()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.add_job(spec)
            for _ in range(60):
                job = app.jobs[0]
                if job.is_terminal:
                    break
                await pilot.pause(0.1)
            assert job.status == DONE
            assert job.output.endswith("result.mp4")
            assert job.percent == 100.0

    asyncio.run(_run())


def test_cancel_running_job(tmp_path, monkeypatch):
    _install_runner(tmp_path, FAKE_RUNNER_SLOW, monkeypatch)
    spec = _make_job_spec(tmp_path)

    async def _run():
        app = PyVideoTransTUI()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.add_job(spec)
            for _ in range(30):
                if app.running is not None and app.running.status == RUNNING:
                    break
                await pilot.pause(0.1)
            assert app.running is not None
            assert app.running.status == RUNNING
            app.action_cancel_job()
            for _ in range(80):
                if app.jobs[0].status == CANCELLED:
                    break
                await pilot.pause(0.1)
            assert app.jobs[0].status == CANCELLED
            assert app.jobs[0].cancel_requested is True

    asyncio.run(_run())
