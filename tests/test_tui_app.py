"""
Smoke tests for the Textual TUI (skipped when textual is not installed).
"""

import asyncio

import pytest

pytest.importorskip("textual")

from tui_app import NewTaskScreen, PyVideoTransTUI


def test_app_mounts_and_has_queue_table():
    async def _run():
        app = PyVideoTransTUI()
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one("#queue-table")
            assert len(table.columns) >= 6
            assert app.query_one("#job-log") is not None
            assert app.query_one("#job-progress") is not None

    asyncio.run(_run())


def test_new_task_screen_composes_with_fake_data():
    async def _run():
        app = PyVideoTransTUI()
        async with app.run_test() as pilot:
            screen = NewTaskScreen(
                [("en", "English"), ("zh-cn", "简体中文")],
                {"recognition": ["faster-whisper"], "translation": ["deepseek"], "tts": ["edge-tts"]},
                {"recogn_type": 0, "translate_type": 0, "tts_type": 0, "model_name": "tiny", "cuda": False, "voice_role": "No"},
            )
            await app.push_screen(screen)
            await pilot.pause()
            assert app.screen.query_one("#f-task") is not None
            assert app.screen.query_one("#f-file") is not None
            assert app.screen.query_one("#f-voice") is not None
            assert app.screen.query_one("#f-advanced") is not None

    asyncio.run(_run())


def test_quit_binding_exits_app():
    async def _run():
        app = PyVideoTransTUI()
        async with app.run_test() as pilot:
            await pilot.press("q")
            await pilot.pause()
            assert app._exit is True

    asyncio.run(_run())
