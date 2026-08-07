"""
Unit tests for the TUI queue helpers (pure logic, no textual/videotrans).
"""

import os

from tui_common import (
    CANCELLED,
    DONE,
    FAILED,
    QUEUED,
    RUNNING,
    Job,
    default_overrides_from_params,
    extract_percent,
    make_output_dir,
    newest_file,
    sanitize_basename,
    stages_for,
)


def test_extract_percent():
    assert extract_percent("识别中 45%") == 45.0
    assert extract_percent("downloading 12.5% of model") == 12.5
    assert extract_percent("no percentage here") is None
    assert extract_percent("0%") == 0.0
    assert extract_percent("120%") == 100.0  # clamped


def test_sanitize_basename():
    assert sanitize_basename("my video.mp4") == "my-video-mp4"
    assert sanitize_basename("a:b?c") == "a-b-c"


def test_make_output_dir_is_unique(tmp_path):
    base = str(tmp_path)
    d1 = make_output_dir("movie", base, ts="20260802_120000")
    d2 = make_output_dir("movie", base, ts="20260802_120001")
    assert d1 != d2
    assert d1.endswith("output/movie_20260802_120000")


def test_newest_file(tmp_path):
    older = tmp_path / "a.txt"
    newer = tmp_path / "b.txt"
    older.write_text("1")
    newer.write_text("2")
    os.utime(older, (1000, 1000))
    os.utime(newer, (2000, 2000))
    assert newest_file(str(tmp_path)) == str(newer)
    empty = tmp_path / "empty"
    empty.mkdir()
    assert newest_file(str(empty)) == ""


def test_job_flow():
    job = Job(job_id=1, task="vtv", name="/tmp/v.mp4", source_lang="zh-cn", target_lang="en")
    assert job.status == QUEUED
    job.status = RUNNING
    assert not job.is_terminal
    job.status = DONE
    assert job.is_terminal
    job.status = FAILED
    assert job.is_terminal
    job.status = CANCELLED
    assert job.is_terminal


def test_job_logs_capped():
    job = Job(job_id=1, task="sts", name="/tmp/s.srt", source_lang="auto", target_lang="en")
    for i in range(300):
        job.add_log(f"line {i}", max_lines=50)
    assert len(job.logs) == 50
    assert job.logs[0] == "line 250"
    assert job.logs[-1] == "line 299"


def test_default_overrides_from_params():
    params = {"recogn_type": 2, "translate_type": 5, "tts_type": 15, "model_name": "large-v3", "cuda": True}
    overrides = default_overrides_from_params(params)
    assert overrides == {
        "recogn_type": 2,
        "translate_type": 5,
        "tts_type": 15,
        "model_name": "large-v3",
        "cuda": True,
    }
    assert default_overrides_from_params({})["model_name"] == "tiny"


def test_stages_for():
    assert [s for s, _ in stages_for("vtv")][:3] == ["prepare", "recogn", "diariz"]
    assert [s for s, _ in stages_for("sts")] == ["prepare", "trans"]
