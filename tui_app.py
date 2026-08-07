"""
tui_app.py -- single-screen dashboard TUI for pyVideoTrans.

Keyboard-first terminal UI (Textual): queue on the left, live progress and
logs for the selected job on the right.  Real work runs in `tui_runner.py`
subprocesses so the UI never blocks and jobs can be cancelled by killing the
process group.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Footer,
    Header,
    Input,
    ProgressBar,
    RichLog,
    Select,
    Static,
)

from tui_common import (
    CANCELLED,
    DONE,
    FAILED,
    QUEUED,
    RUNNING,
    STAGE_LABEL,
    STATUS_LABEL,
    TASK_LABEL,
    Job,
    default_overrides_from_params,
)
from tui_data import (
    ROOT_DIR,
    ensure_output_dir,
    list_languages,
    list_providers,
    provider_name,
    read_params,
    voice_roles,
)


class NewTaskScreen(ModalScreen[Optional[dict]]):
    """Modal form for adding a vtv/sts job to the queue."""

    BINDINGS = [Binding("escape", "cancel", "取消")]

    def __init__(
        self,
        languages: list[tuple[str, str]],
        providers: dict[str, list[str]],
        defaults: dict,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._languages = languages
        self._providers = providers
        self._defaults = defaults
        self._advanced_visible = False
        self._voice_options: list[tuple[str, str]] = [("No", "No")]

    def compose(self) -> ComposeResult:
        src_default = self._defaults.get("source_language", "auto")
        tgt_default = self._defaults.get("target_language", "zh-cn")
        lang_codes = [code for code, _ in self._languages]
        src_options = [(name, code) for code, name in self._languages]
        tgt_options = [(name, code) for code, name in self._languages]
        yield Static("新建任务", id="form-title")
        with Vertical(id="form-body"):
            yield Select(
                [("视频翻译 (VTV)", "vtv"), ("字幕翻译 (STS)", "sts")],
                value="vtv",
                id="f-task",
                prompt="任务类型",
            )
            yield Input(
                placeholder="输入文件绝对路径，如 /path/to/video.mp4",
                id="f-file",
            )
            yield Select(
                src_options,
                value=src_default if src_default in lang_codes else src_options[0][1],
                id="f-src",
                prompt="源语言",
            )
            yield Select(
                tgt_options,
                value=tgt_default if tgt_default in lang_codes else tgt_options[0][1],
                id="f-tgt",
                prompt="目标语言",
            )
            yield Select(self._voice_options, id="f-voice", prompt="音色")
            yield Button("高级选项", id="f-advanced-btn", variant="default")
            with Vertical(id="f-advanced", classes="hidden"):
                yield Input(
                    value=str(self._defaults.get("model_name", "tiny")),
                    placeholder="识别模型 (如 large-v3)",
                    id="f-model",
                )
                yield Select(
                    [
                        (provider_name(self._providers, "translation", i), i)
                        for i in range(len(self._providers.get("translation", [])))
                    ]
                    or [("默认", 0)],
                    value=min(
                        int(self._defaults.get("translate_type", 0)),
                        max(len(self._providers.get("translation", [])) - 1, 0),
                    ),
                    id="f-trans",
                    prompt="翻译渠道",
                )
                yield Select(
                    [
                        (provider_name(self._providers, "tts", i), i)
                        for i in range(len(self._providers.get("tts", [])))
                    ]
                    or [("默认", 0)],
                    value=min(
                        int(self._defaults.get("tts_type", 0)),
                        max(len(self._providers.get("tts", [])) - 1, 0),
                    ),
                    id="f-tts",
                    prompt="配音引擎",
                )
                yield Checkbox(
                    "启用 CUDA 加速",
                    value=bool(self._defaults.get("cuda", False)),
                    id="f-cuda",
                )
            with Horizontal(id="form-actions"):
                yield Button("提交", id="f-submit", variant="primary")
                yield Button("取消", id="f-cancel", variant="error")

    def on_mount(self) -> None:
        self._refresh_voice()

    def _selected(self, widget_id: str):
        w = self.query_one(f"#{widget_id}", Select)
        return w.value

    def _current_tts(self) -> int:
        try:
            return int(self._selected("f-tts") or self._defaults.get("tts_type", 0))
        except (TypeError, ValueError):
            return 0

    def _current_lang(self) -> str:
        return str(self._selected("f-tgt") or "en")

    def _refresh_voice(self) -> None:
        tts_type = self._current_tts()
        self._voice_options = voice_roles(tts_type, self._current_lang())
        w = self.query_one("#f-voice", Select)
        current = w.value
        w.set_options(self._voice_options)
        codes = [code for _, code in self._voice_options]
        if current in codes:
            w.value = current
        elif self._defaults.get("voice_role") in codes:
            w.value = self._defaults["voice_role"]
        else:
            w.value = "No"

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id in ("f-tgt", "f-tts"):
            self._refresh_voice()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "f-advanced-btn":
            self._advanced_visible = not self._advanced_visible
            self.query_one("#f-advanced").set_class(self._advanced_visible, "visible")
            event.button.label = "高级选项 (隐藏)" if self._advanced_visible else "高级选项"
        elif event.button.id == "f-cancel":
            self.dismiss(None)
        elif event.button.id == "f-submit":
            self._submit()

    def _submit(self) -> None:
        file_input = self.query_one("#f-file", Input)
        path = file_input.value.strip()
        if not path or not Path(path).is_file():
            self.notify("文件不存在，请检查路径", severity="error", timeout=5)
            return
        task = str(self._selected("f-task") or "vtv")
        target = str(self._selected("f-tgt") or "")
        if not target:
            self.notify("请选择目标语言", severity="error", timeout=5)
            return
        source = str(self._selected("f-src") or "auto")
        if task == "vtv" and (not source or source == "auto"):
            self.notify("视频翻译 (VTV) 需要明确选择源语言", severity="error", timeout=5)
            return
        voice = str(self._selected("f-voice") or "No")
        if task == "vtv" and not voice:
            self.notify("请选择音色", severity="error", timeout=5)
            return
        overrides = {
            "recogn_type": int(self._defaults.get("recogn_type", 0)),
            "translate_type": int(self._selected("f-trans") or 0),
            "tts_type": int(self._selected("f-tts") or 0),
            "model_name": (self.query_one("#f-model", Input).value or "tiny").strip(),
            "cuda": bool(self.query_one("#f-cuda", Checkbox).value),
        }
        self.dismiss(
            {
                "task": task,
                "name": str(Path(path).resolve()),
                "source_language_code": source,
                "target_language_code": target,
                "voice_role": voice,
                "overrides": overrides,
            }
        )

    def action_cancel(self) -> None:
        self.dismiss(None)


class PyVideoTransTUI(App):
    """Single-screen dashboard for vtv/sts batch jobs."""

    TITLE = "pyVideoTrans TUI"
    SUB_TITLE = "视频翻译 / 字幕翻译 · 批处理队列"
    BINDINGS = [
        Binding("n", "new_task", "新建任务"),
        Binding("c", "cancel_job", "取消"),
        Binding("r", "retry_job", "重试"),
        Binding("x", "remove_job", "移除"),
        Binding("q", "quit", "退出"),
    ]
    CSS = """
    #queue-pane { width: 46%; border-right: solid $primary; padding: 0 1; }
    #detail-pane { width: 54%; padding: 0 1; }
    #queue-table { height: 1fr; }
    .pane-title { text-style: bold; color: $accent; margin-bottom: 1; }
    .muted { color: $text-muted; }
    #job-info { height: auto; margin-bottom: 1; }
    #job-stage { margin-top: 1; margin-bottom: 1; }
    #job-log { height: 1fr; border: round $surface; }
    #form-body { height: auto; padding: 0 2; }
    #form-title { text-style: bold; text-align: center; padding: 1; }
    #f-advanced { display: none; }
    #f-advanced.visible { display: block; }
    #form-actions { height: 3; align-horizontal: right; padding-top: 1; }
    .hidden { display: none; }
    """

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="queue-pane"):
                yield Static("任务队列", classes="pane-title")
                yield DataTable(id="queue-table")
                yield Static(
                    "n 新建 · c 取消 · r 重试 · x 移除 · q 退出",
                    id="queue-hint",
                    classes="muted",
                )
            with Vertical(id="detail-pane"):
                yield Static("任务详情", classes="pane-title")
                yield Static("未选择任务", id="job-info")
                yield ProgressBar(total=100, show_eta=False, id="job-progress")
                yield Static("", id="job-stage", classes="muted")
                yield RichLog(id="job-log", highlight=False, markup=False, wrap=True)
        yield Footer()

    def __init__(self) -> None:
        super().__init__()
        self.jobs: list[Job] = []
        self.queue: list[Job] = []
        self.running: Optional[Job] = None
        self._job_seq = 0
        self._selected_id: Optional[int] = None
        self._languages: list[tuple[str, str]] = []
        self._providers: dict[str, list[str]] = {}
        self._defaults: dict = {}
        self._data_ready = False
        self._col_keys = None

    def on_mount(self) -> None:
        # Compose content mounts after App.on_mount in modern Textual, so defer
        # widget setup until the first render cycle.
        self.call_after_refresh(self._setup_table)
        self.run_worker(self._load_data, name="tui-data-loader")
        self.run_worker(self._dispatcher_loop, name="tui-dispatcher")
        self.set_interval(0.5, self._tick)

    def _setup_table(self) -> None:
        table = self.query_one("#queue-table", DataTable)
        if self._col_keys is None:
            self._col_keys = table.add_columns("ID", "任务", "文件", "状态", "阶段", "进度")
        table.cursor_type = "row"
        table.zebra_stripes = True
        self.refresh_table()

    async def _load_data(self) -> None:
        langs, provs, params = await asyncio.gather(
            asyncio.to_thread(list_languages),
            asyncio.to_thread(list_providers),
            asyncio.to_thread(read_params),
        )
        self._languages = langs
        self._providers = provs
        self._defaults = default_overrides_from_params(params)
        self._defaults["voice_role"] = params.get("voice_role") or "No"
        self._defaults["source_language"] = params.get("source_language") or "auto"
        self._defaults["target_language"] = params.get("target_language") or "zh-cn"
        self._data_ready = True

    def _tick(self) -> None:
        if self.running is None:
            return
        self.refresh_detail()
        self._update_running_row()

    def _update_running_row(self) -> None:
        if self.running is None or self._col_keys is None:
            return
        try:
            table = self.query_one("#queue-table", DataTable)
            stage = STAGE_LABEL.get(self.running.stage, self.running.stage or "")
            table.update_cell(
                str(self.running.job_id),
                self._col_keys[4],
                stage,
            )
            table.update_cell(
                str(self.running.job_id),
                self._col_keys[5],
                f"{self.running.percent:.0f}%",
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Queue model
    # ------------------------------------------------------------------
    def _new_job(self, spec: dict) -> Job:
        self._job_seq += 1
        return Job(
            job_id=self._job_seq,
            task=spec["task"],
            name=spec["name"],
            source_lang=spec["source_language_code"],
            target_lang=spec["target_language_code"],
            voice_role=spec.get("voice_role", "No"),
            overrides=spec.get("overrides", {}),
        )

    def add_job(self, spec: dict) -> None:
        job = self._new_job(spec)
        self.jobs.append(job)
        self.queue.append(job)
        self.refresh_table()
        self.notify(f"任务 #{job.job_id} 已加入队列", timeout=3)

    def selected(self) -> Optional[Job]:
        for job in self.jobs:
            if job.job_id == self._selected_id:
                return job
        return None

    def refresh_table(self) -> None:
        if self._col_keys is None:
            return
        table = self.query_one("#queue-table", DataTable)
        table.clear()
        for job in self.jobs:
            percent = f"{job.percent:.0f}%" if job.status == RUNNING else ""
            table.add_row(
                str(job.job_id),
                TASK_LABEL.get(job.task, job.task),
                job.display_name,
                STATUS_LABEL.get(job.status, job.status),
                STAGE_LABEL.get(job.stage, job.stage or ""),
                percent,
                key=str(job.job_id),
            )
        self.refresh_detail()

    def refresh_detail(self) -> None:
        job = self.selected() or self.running
        info = self.query_one("#job-info", Static)
        stage_w = self.query_one("#job-stage", Static)
        bar = self.query_one("#job-progress", ProgressBar)
        log = self.query_one("#job-log", RichLog)
        if job is None:
            info.update("未选择任务")
            stage_w.update("")
            bar.update(progress=0)
            log.clear()
            return
        role = job.voice_role if job.task == "vtv" else "-"
        info.update(
            f"[bold]#{job.job_id} {TASK_LABEL.get(job.task, job.task)}[/]\n"
            f"文件: {job.name}\n"
            f"语言: {job.source_lang} → {job.target_lang}\n"
            f"音色: {role}\n"
            f"状态: {STATUS_LABEL.get(job.status, job.status)}"
            + (f"\n输出: {job.output}" if job.output else "")
            + (f"\n错误: {job.message}" if job.message else "")
        )
        stage = STAGE_LABEL.get(job.stage, job.stage or "")
        stage_w.update(f"阶段: {stage or '—'}  进度: {job.percent:.0f}%" if job.status == RUNNING else "")
        bar.update(progress=job.percent if job.status == RUNNING else 100 if job.status == DONE else 0)
        log.clear()
        for line in job.logs[-100:]:
            log.write(line)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        try:
            self._selected_id = int(event.row_key.value)
        except (TypeError, ValueError):
            return
        self.refresh_detail()

    # ------------------------------------------------------------------
    # Job execution (serial, one subprocess per job)
    # ------------------------------------------------------------------
    async def _dispatcher_loop(self) -> None:
        while True:
            if self.running is None and self.queue:
                job = self.queue.pop(0)
                self.running = job
                job.status = RUNNING
                self._selected_id = job.job_id
                self.refresh_table()
                await self._run_job(job)
                self.running = None
                self.refresh_table()
            else:
                await asyncio.sleep(0.2)

    def _job_spec_file(self, job: Job) -> str:
        ts = time.strftime("%Y%m%d_%H%M%S")
        return os.path.join(tempfile.gettempdir(), f"tui_job_{job.job_id}_{ts}.json")

    async def _run_job(self, job: Job) -> None:
        noext = Path(job.name).stem
        job.output_dir = ensure_output_dir(str(ROOT_DIR), noext)
        spec = {
            "task": job.task,
            "name": job.name,
            "source_language_code": job.source_lang,
            "target_language_code": job.target_lang,
            "voice_role": job.voice_role,
            "output_dir": job.output_dir,
            "overrides": job.overrides,
        }
        spec_file = self._job_spec_file(job)
        Path(spec_file).write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(ROOT_DIR / "tui_runner.py"),
            spec_file,
            cwd=str(ROOT_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        job.proc = proc
        job.message = ""
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                try:
                    evt = json.loads(text)
                except json.JSONDecodeError:
                    job.add_log(text)
                    continue
                self._handle_event(job, evt)
            await proc.wait()
        finally:
            try:
                os.unlink(spec_file)
            except OSError:
                pass
            job.proc = None
            if job.status == RUNNING:
                job.status = CANCELLED if job.cancel_requested else FAILED
                job.message = "任务已取消" if job.status == CANCELLED else "进程异常退出"

    def _handle_event(self, job: Job, evt: dict) -> None:
        kind = evt.get("event")
        if kind == "stage":
            job.stage = evt.get("stage", "")
            if evt.get("state") == "start":
                job.percent = 0.0
        elif kind == "log":
            job.add_log(evt.get("text", ""))
            if evt.get("percent") is not None:
                job.percent = float(evt["percent"])
        elif kind == "cancelling":
            job.cancel_requested = True
            job.message = "正在取消…"
        elif kind == "cancelled":
            job.status = CANCELLED
            job.message = "任务已取消"
        elif kind == "done":
            job.status = DONE
            job.output = evt.get("output", "") or evt.get("output_dir", "")
            job.percent = 100.0
        elif kind == "error":
            job.status = FAILED
            job.message = evt.get("message", "未知错误")
            job.add_log(f"错误: {job.message}")

    def _signal_group(self, proc, sig: signal.Signals) -> None:
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    async def _cancel_async(self, job: Job) -> None:
        self._signal_group(job.proc, signal.SIGTERM)
        try:
            await asyncio.wait_for(job.proc.wait(), timeout=10)
        except asyncio.TimeoutError:
            self._signal_group(job.proc, signal.SIGKILL)
        if job.status == RUNNING:
            job.status = CANCELLED
            job.message = "任务已取消"
            job.add_log("任务已取消")
            self.refresh_table()

    # ------------------------------------------------------------------
    # Actions (keyboard-first)
    # ------------------------------------------------------------------
    def action_new_task(self) -> None:
        if not self._data_ready:
            self.notify("语言/渠道数据加载中，请稍候…", timeout=3)
            return
        screen = NewTaskScreen(self._languages, self._providers, self._defaults)
        self.push_screen(screen, self._on_task_form_dismissed)

    def _on_task_form_dismissed(self, spec: Optional[dict]) -> None:
        if spec:
            self.add_job(spec)

    def action_cancel_job(self) -> None:
        job = self.running
        if job is None or job.status != RUNNING or job.proc is None:
            self.notify("没有正在运行的任务", timeout=3)
            return
        job.cancel_requested = True
        job.message = "正在取消…"
        self.run_worker(self._cancel_async(job), name=f"cancel-{job.job_id}")

    def action_retry_job(self) -> None:
        job = self.selected() or self.running
        if job is None or job.status not in (FAILED, CANCELLED):
            self.notify("请选择一个失败或已取消的任务", timeout=3)
            return
        job.status = QUEUED
        job.stage = ""
        job.percent = 0.0
        job.message = ""
        job.output = ""
        job.output_dir = ""
        job.logs.clear()
        self.queue.append(job)
        self.notify(f"任务 #{job.job_id} 已重新排队", timeout=3)
        self.refresh_table()

    def action_remove_job(self) -> None:
        job = self.selected()
        if job is None or job.status == RUNNING or job in self.queue:
            self.notify("只能移除已结束的任务", timeout=3)
            return
        self.jobs.remove(job)
        self._selected_id = None
        self.refresh_table()

    def action_quit(self) -> None:
        if self.running is not None and self.running.proc is not None:
            self._signal_group(self.running.proc, signal.SIGTERM)
        self.exit()


def main() -> None:
    PyVideoTransTUI().run()


if __name__ == "__main__":
    main()
