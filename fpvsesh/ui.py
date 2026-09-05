"""Local desktop controls for the same CLI used by unattended FPV Sesh renders."""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


APP_DIR = Path(__file__).resolve().parents[1]
MEDIA_TYPES = [("Video recordings", "*.mp4 *.mov *.mkv *.m4v *.avi *.mts *.m2ts"), ("All files", "*.*")]
STYLES = {"Sesh Hype": "hype", "Cinematic Flow": "cinematic", "Freestyle Focus": "freestyle"}
LOOKS = {"FPV Punch": "punch", "Natural": "natural", "Cinematic": "cinematic"}
QUALITIES = {"Auto": "auto", "Lanczos": "lanczos", "AI (tested backend)": "ai"}
DURATIONS = {"Auto": "auto", "30 seconds": "30", "60 seconds": "60", "90 seconds": "90", "120 seconds": "120"}
FINISH_MODES = ("Preview, then final automatically", "Preview only", "Preview, then approve final")


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return default


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def timestamp(seconds) -> str:
    try:
        value = max(0, float(seconds))
    except (ValueError, TypeError):
        return "?"
    minutes, rest = divmod(value, 60)
    return f"{int(minutes):02d}:{rest:05.2f}"


class SeshApp(tk.Tk):
    def __init__(self, app_dir: Path = APP_DIR):
        super().__init__()
        self.app_dir = app_dir
        self.title("FPV Sesh")
        self.geometry("1180x880")
        self.minsize(1050, 800)
        self.configure(background="#eef2f7")
        self.events: queue.Queue = queue.Queue()
        self.process: subprocess.Popen | None = None
        self.job_dir: Path | None = None
        self.last_args: list[str] | None = None
        self.files: list[Path] = []
        self.folder: Path | None = None
        self.candidates: list[dict] = []
        self.override_choices = {"keep": [], "exclude": []}
        self.warnings_seen: set[str] = set()
        self.paused = False
        self.closing = False
        self.overrides_dirty = False
        self.run_preview_only = False
        self.terminal_stage = ""
        self._candidate_mtime = None
        self._last_diagnostics = None
        self.settings_widgets: list[tuple[tk.Widget, str]] = []
        self._make_styles()
        self._build()
        self._seed_inputs()
        self._refresh_diagnostics()
        self._load_recent_job()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.after(120, self._poll)

    def _make_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#eef2f7")
        style.configure("TLabel", background="#eef2f7", foreground="#17243a", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 25, "bold"))
        style.configure("Muted.TLabel", foreground="#526278", font=("Segoe UI", 9))
        style.configure("TLabelframe", background="#eef2f7", bordercolor="#ccd5e0")
        style.configure("TLabelframe.Label", background="#eef2f7", foreground="#17243a", font=("Segoe UI", 10, "bold"))
        style.configure("TButton", padding=(10, 7), font=("Segoe UI", 9))
        style.configure("Primary.TButton", background="#1166c3", foreground="#ffffff", font=("Segoe UI", 11, "bold"))
        style.map("Primary.TButton", background=[("active", "#0c509c"), ("disabled", "#a5b5c7")])
        style.configure("TCheckbutton", background="#eef2f7")
        style.configure("Treeview", rowheight=28, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))
        style.configure("TNotebook", background="#eef2f7")

    def _build(self):
        shell = ttk.Frame(self, padding=20)
        shell.pack(fill="both", expand=True)
        header = ttk.Frame(shell)
        header.pack(fill="x")
        ttk.Label(header, text="FPV Sesh", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="Your footage. One session edit. Runs locally.", style="Muted.TLabel").pack(side="left", padx=20)
        self.gpu_text = tk.StringVar(value="GPU: diagnostics pending")
        ttk.Label(shell, textvariable=self.gpu_text, style="Muted.TLabel", wraplength=1120).pack(anchor="w", pady=(4, 12))

        upper = ttk.Frame(shell)
        upper.pack(fill="x")
        upper.columnconfigure(0, weight=2)
        upper.columnconfigure(1, weight=3)

        inputs = ttk.LabelFrame(upper, text="1  Choose your session", padding=12)
        inputs.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self.input_list = tk.Listbox(inputs, height=7, selectmode="extended", activestyle="none", borderwidth=1,
                                    relief="solid", background="white", foreground="#17243a", font=("Segoe UI", 9))
        self.input_list.pack(fill="both", expand=True)
        input_buttons = ttk.Frame(inputs)
        input_buttons.pack(fill="x", pady=(8, 0))
        self.add_button = ttk.Button(input_buttons, text="Add clips", command=self._choose_files)
        self.add_button.pack(side="left")
        self.folder_button = ttk.Button(input_buttons, text="Choose folder", command=self._choose_folder)
        self.folder_button.pack(side="left", padx=5)
        self.remove_button = ttk.Button(input_buttons, text="Remove", command=self._remove_files)
        self.remove_button.pack(side="left")
        self.input_text = tk.StringVar()
        input_description = ttk.Label(inputs, textvariable=self.input_text, style="Muted.TLabel", wraplength=410)
        input_description.pack(anchor="w", pady=(6, 0))
        input_note = ttk.Label(inputs, text="Original files stay untouched. Music is off for this version.",
                               style="Muted.TLabel", wraplength=410)
        input_note.pack(anchor="w", pady=(6, 0))
        inputs.bind("<Configure>", lambda event: [label.configure(wraplength=max(200, event.width - 28)) for label in (input_description, input_note)])

        options = ttk.LabelFrame(upper, text="2  Shape your edit", padding=12)
        options.grid(row=0, column=1, sticky="nsew")
        for col in (1, 3):
            options.columnconfigure(col, weight=1)
        self.style_value = tk.StringVar(value="Sesh Hype")
        self.duration_value = tk.StringVar(value="Auto")
        self.look_value = tk.StringVar(value="FPV Punch")
        self.quality_value = tk.StringVar(value="Auto")
        self.codec_value = tk.StringVar(value="HEVC")
        self.finish_value = tk.StringVar(value=FINISH_MODES[0])
        self.strength_value = tk.DoubleVar(value=0.55)
        self.audio_value = tk.DoubleVar(value=0.4)
        self._combo(options, 0, 0, "Style", self.style_value, list(STYLES))
        self._combo(options, 0, 2, "Duration", self.duration_value, list(DURATIONS))
        self._combo(options, 1, 0, "Color look", self.look_value, list(LOOKS))
        self.quality_combo = self._combo(options, 1, 2, "Enhancement", self.quality_value, ["Auto", "Lanczos"])
        self._slider(options, 2, "Color strength", self.strength_value)
        self._slider(options, 3, "Source sound", self.audio_value)
        self._combo(options, 4, 0, "Export", self.codec_value, ["HEVC", "H.264"])
        self._combo(options, 4, 2, "Finish", self.finish_value, FINISH_MODES, width=29)
        self.quality_note = tk.StringVar(value="Auto uses conventional GPU scaling when tested, with a Lanczos fallback.")
        quality_note = ttk.Label(options, textvariable=self.quality_note, style="Muted.TLabel", wraplength=600)
        quality_note.grid(row=5, column=0, columnspan=4, sticky="w", pady=(8, 0))
        options.bind("<Configure>", lambda event: quality_note.configure(wraplength=max(300, event.width - 28)))
        gyro_row = ttk.Frame(options)
        gyro_row.grid(row=6, column=0, columnspan=4, sticky="w", pady=(5, 0))
        ttk.Checkbutton(gyro_row, text="Gyro stabilization", state="disabled").pack(side="left")
        ttk.Label(gyro_row, text="Off: requires validated gyro data and lens calibration.", style="Muted.TLabel").pack(side="left", padx=4)

        actions = ttk.Frame(shell)
        actions.pack(fill="x", pady=(15, 8))
        self.make_button = ttk.Button(actions, text="Make My Sesh", style="Primary.TButton", command=self._start_new)
        self.make_button.pack(side="left")
        self.pause_button = ttk.Button(actions, text="Pause", command=self._pause, state="disabled")
        self.pause_button.pack(side="left", padx=(8, 4))
        self.cancel_button = ttk.Button(actions, text="Cancel", command=self._cancel, state="disabled")
        self.cancel_button.pack(side="left")
        self.resume_button = ttk.Button(actions, text="Resume saved job…", command=self._resume_job)
        self.resume_button.pack(side="left", padx=8)
        self.play_final_button = ttk.Button(actions, text="Play final 4K", command=self._play_final, state="disabled")
        self.play_final_button.pack(side="right")
        self.preview_button = ttk.Button(actions, text="Play edit preview (720p)", command=self._play_preview, state="disabled")
        self.preview_button.pack(side="right", padx=(0, 6))
        ttk.Button(actions, text="Open output", command=self._open_output).pack(side="right", padx=6)

        self.stage_text = tk.StringVar(value="Ready — choose clips and make your session edit.")
        ttk.Label(shell, textvariable=self.stage_text, wraplength=1110).pack(anchor="w", pady=(2, 4))
        self.progress = ttk.Progressbar(shell, maximum=100, mode="determinate")
        self.progress.pack(fill="x")
        self.detail_text = tk.StringVar(value="Progress comes from the renderer. Pause and cancel take effect at supported stage or segment boundaries.")
        ttk.Label(shell, textvariable=self.detail_text, style="Muted.TLabel", wraplength=1120).pack(anchor="w", pady=(4, 10))

        notebook = ttk.Notebook(shell)
        notebook.pack(fill="both", expand=True)
        review = ttk.Frame(notebook, padding=10)
        log = ttk.Frame(notebook, padding=10)
        notebook.add(review, text="Review moments")
        notebook.add(log, text="Progress & warnings")
        self.review_text = tk.StringVar(value="Analyzed moments will appear here. Select rows to keep or exclude, then regenerate the edit.")
        ttk.Label(review, textvariable=self.review_text, style="Muted.TLabel", wraplength=1100).pack(anchor="w", pady=(0, 7))
        table_frame = ttk.Frame(review)
        table_frame.pack(fill="both", expand=True)
        columns = ("choice", "source", "start", "end", "score", "reason")
        self.table = ttk.Treeview(table_frame, columns=columns, show="headings", selectmode="extended", height=7)
        for col, title, width in (("choice", "Choice", 90), ("source", "Source recording", 145), ("start", "In", 72),
                                  ("end", "Out", 72), ("score", "Score", 62), ("reason", "Reason / estimated signals", 480)):
            self.table.heading(col, text=title)
            self.table.column(col, width=width, minwidth=50, stretch=(col in ("source", "reason")))
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scrollbar.set)
        self.table.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.table.bind("<Double-1>", self._show_candidate)
        review_actions = ttk.Frame(review)
        table_frame.pack_forget()
        review_actions.pack(side="bottom", fill="x", pady=(8, 0))
        table_frame.pack(fill="both", expand=True)
        self.keep_button = ttk.Button(review_actions, text="Keep selected", command=lambda: self._set_choice("keep"), state="disabled")
        self.keep_button.pack(side="left")
        self.exclude_button = ttk.Button(review_actions, text="Exclude selected", command=lambda: self._set_choice("exclude"), state="disabled")
        self.exclude_button.pack(side="left", padx=5)
        self.reset_button = ttk.Button(review_actions, text="Clear overrides", command=self._reset_choices, state="disabled")
        self.reset_button.pack(side="left")
        self.regenerate_button = ttk.Button(review_actions, text="Regenerate edit", command=self._regenerate, state="disabled")
        self.regenerate_button.pack(side="right")
        self.final_button = ttk.Button(review_actions, text="Render final 4K", command=self._render_final, state="disabled")
        self.final_button.pack(side="right", padx=6)
        self.log_box = tk.Text(log, height=12, wrap="word", state="disabled", font=("Consolas", 9), background="white", relief="flat")
        self.log_box.pack(fill="both", expand=True)
        notebook.pack_forget()
        ttk.Label(shell, text="Playback opens your Windows video player. A render report and exact source timeline are saved with each edit.",
                  style="Muted.TLabel", wraplength=1120).pack(side="bottom", anchor="w", pady=(10, 0))
        notebook.pack(fill="both", expand=True)

    def _combo(self, parent, row, column, label, variable, values, width=19):
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", padx=(0 if column == 0 else 12, 7), pady=4)
        widget = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly", width=width)
        widget.grid(row=row, column=column + 1, sticky="ew", pady=4)
        self.settings_widgets.append((widget, "readonly"))
        return widget

    def _slider(self, parent, row, label, variable):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5)
        scale = ttk.Scale(parent, from_=0, to=1, variable=variable)
        scale.grid(row=row, column=1, columnspan=2, sticky="ew", padx=(0, 8))
        value_text = tk.StringVar(value=f"{variable.get():.0%}")
        variable.trace_add("write", lambda *_: value_text.set(f"{variable.get():.0%}"))
        ttk.Label(parent, textvariable=value_text).grid(row=row, column=3, sticky="w")
        self.settings_widgets.append((scale, "normal"))

    def _seed_inputs(self):
        supplied = Path.home() / "Downloads"
        self.files = [supplied / f"DJIU000{index}.mp4" for index in range(3) if (supplied / f"DJIU000{index}.mp4").is_file()]
        if not self.files:
            self.folder = self.app_dir / "input"
        self._refresh_inputs()

    def _refresh_inputs(self):
        self.input_list.delete(0, "end")
        if self.folder:
            self.input_list.insert("end", f"Folder: {self.folder}")
            self.input_text.set("The selected folder will be inspected when the job starts.")
        else:
            for path in self.files:
                self.input_list.insert("end", str(path))
            self.input_text.set(f"{len(self.files)} recording(s) selected. Full recordings are analyzed before highlights are chosen.")

    def _choose_files(self):
        selected = filedialog.askopenfilenames(parent=self, title="Choose original session recordings", filetypes=MEDIA_TYPES)
        if selected:
            self.folder = None
            self.files = list(dict.fromkeys([*self.files, *(Path(item) for item in selected)]))
            self._refresh_inputs()

    def _choose_folder(self):
        folder = filedialog.askdirectory(parent=self, title="Choose a folder of session recordings", initialdir=self.app_dir / "input")
        if folder:
            self.folder, self.files = Path(folder), []
            self._refresh_inputs()

    def _remove_files(self):
        selected = self.input_list.curselection()
        if self.folder and selected:
            self.folder = None
        else:
            self.files = [path for index, path in enumerate(self.files) if index not in selected]
        self._refresh_inputs()

    def _command_args(self) -> list[str]:
        args = ["make"]
        if self.folder:
            args += ["--folder", str(self.folder)]
        else:
            for path in self.files:
                args += ["--input", str(path)]
        args += ["--duration", DURATIONS[self.duration_value.get()], "--style", STYLES[self.style_value.get()],
                 "--look", LOOKS[self.look_value.get()], "--strength", f"{self.strength_value.get():.4f}",
                 "--quality", QUALITIES[self.quality_value.get()], "--audio-level", f"{self.audio_value.get():.4f}",
                 "--codec", "hevc" if self.codec_value.get() == "HEVC" else "h264"]
        if self.finish_value.get() != FINISH_MODES[0]:
            args.append("--preview-only")
        return args

    @staticmethod
    def _without_options(args, names):
        cleaned = []
        index = 0
        while index < len(args):
            item = args[index]
            if item in names:
                index += 1 if item in ("--preview-only", "--regenerate") else 2
            else:
                cleaned.append(item)
                index += 1
        return cleaned

    def _start_new(self):
        if not self.folder and not self.files:
            messagebox.showinfo("Choose recordings", "Add one or more recordings or choose your session folder.", parent=self)
            return
        self.job_dir = None
        self._candidate_mtime = None
        self.candidates = []
        self.override_choices = {"keep": [], "exclude": []}
        self.overrides_dirty = False
        self._paint_candidates()
        self.last_args = self._command_args()
        self._launch(self.last_args)

    def _launch(self, args):
        if self.process and self.process.poll() is None:
            return
        self.paused = False
        self.run_preview_only = "--preview-only" in args
        self.terminal_stage = ""
        self.pause_button.configure(text="Pause")
        self.progress["value"] = 0
        self.stage_text.set("Starting the local renderer…")
        self.detail_text.set("Preparing this job. Original footage stays untouched.")
        executable = self.app_dir / ".venv" / "Scripts" / "python.exe"
        if not executable.is_file():
            executable = Path(sys.executable)
        command = [str(executable), "-m", "fpvsesh.cli", *args]
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"
        try:
            self.process = subprocess.Popen(command, cwd=self.app_dir, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                            env=environment, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as exc:
            self.process = None
            self.stage_text.set("The renderer could not start.")
            self._log(str(exc))
            messagebox.showerror("Could not start FPV Sesh", str(exc), parent=self)
            return
        self._set_busy(True)
        self._log("Started a local render job.")
        threading.Thread(target=self._read_process, args=(self.process,), daemon=True).start()

    def _read_process(self, process):
        try:
            if process.stdout:
                for line in process.stdout:
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                    except ValueError:
                        payload = {"message": line.rstrip()}
                    self.events.put(("event", payload if isinstance(payload, dict) else {"message": str(payload)}))
        finally:
            self.events.put(("exit", process.wait()))

    def _poll(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "event":
                    self._on_event(payload)
                elif kind == "exit":
                    self.process = None
                    self._set_busy(False)
                    self._load_candidates(force=True)
                    self._refresh_outputs()
                    if self.terminal_stage == "cancelled":
                        self.stage_text.set("Cancelled safely — completed work is kept. Resume the saved job when ready.")
                    elif payload == 0:
                        self.progress["value"] = 100
                        if not self.run_preview_only and self.job_dir and (self.job_dir / "final_4k.mp4").is_file():
                            self.stage_text.set("Your session edit is ready — final 4K video and preview are saved.")
                        elif self.job_dir and (self.job_dir / "preview.mp4").is_file():
                            self.stage_text.set("Preview ready — review your moments or choose Render final 4K.")
                        else:
                            self.stage_text.set("Job finished. Check the run report and messages for its result.")
                    else:
                        self.stage_text.set("Job stopped — completed work is kept. Read the messages or resume the saved job.")
                    self._log(f"Renderer exited with code {payload}.")
                    if self.closing:
                        self.destroy()
                        return
        except queue.Empty:
            pass
        self._load_candidates()
        self._refresh_diagnostics()
        self._refresh_outputs()
        self.after(150, self._poll)

    def _on_event(self, payload):
        job = payload.get("job") or payload.get("job_dir")
        if job:
            path = Path(job)
            self.job_dir = path if path.is_absolute() else self.app_dir / path
        stage = str(payload.get("stage", ""))
        if stage in ("complete", "cancelled", "error"):
            self.terminal_stage = stage
        message = str(payload.get("message", ""))
        if stage:
            self.stage_text.set(f"{stage.replace('_', ' ').capitalize()}: {message}" if message else stage.replace("_", " ").capitalize())
        elif message:
            self.detail_text.set(message)
        if "progress" in payload:
            try:
                self.progress["value"] = min(100, max(0, float(payload["progress"]) * 100))
            except (TypeError, ValueError):
                pass
        if message:
            self._log((stage + ": " if stage else "") + message)
        for warning in payload.get("warnings", []):
            self._log_warning(str(warning))

    def _log(self, text):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text + "\n")
        if int(self.log_box.index("end-1c").split(".")[0]) > 1000:
            self.log_box.delete("1.0", "201.0")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _log_warning(self, text):
        if text not in self.warnings_seen:
            self.warnings_seen.add(text)
            self._log("Warning: " + text)

    def _set_busy(self, busy):
        for button in (self.make_button, self.add_button, self.folder_button, self.remove_button, self.resume_button):
            button.configure(state="disabled" if busy else "normal")
        for widget, idle_state in self.settings_widgets:
            widget.configure(state="disabled" if busy else idle_state)
        self.pause_button.configure(state="normal" if busy else "disabled")
        self.cancel_button.configure(state="normal" if busy else "disabled")
        self._refresh_outputs()

    def _pause(self):
        self.paused = not self.paused
        action = "pause" if self.paused else "resume"
        write_json(self.app_dir / "cache" / "control.json", {"action": action})
        self.pause_button.configure(text="Resume" if self.paused else "Pause")
        self.detail_text.set("Pause requested — the current supported stage or segment will finish first." if self.paused
                             else "Resume requested — processing will continue from the current checkpoint.")
        self._log(self.detail_text.get())

    def _cancel(self):
        write_json(self.app_dir / "cache" / "control.json", {"action": "cancel"})
        self.cancel_button.configure(state="disabled")
        self.pause_button.configure(state="disabled")
        self.detail_text.set("Cancel requested — waiting for a safe boundary. Completed outputs and cached segments are retained.")
        self._log(self.detail_text.get())

    def _resume_job(self):
        selected = filedialog.askdirectory(parent=self, title="Choose a saved job folder", initialdir=self.app_dir / "output")
        if not selected:
            return
        job = Path(selected)
        if not (job / "status.json").is_file() and not (job / "timeline.json").is_file():
            messagebox.showinfo("Choose a job folder", "Select the individual output folder containing status.json or timeline.json.", parent=self)
            return
        self.job_dir = job
        self._candidate_mtime = None
        self.last_args = ["make", "--job", str(job)]
        self._restore_settings(job)
        self._load_candidates(force=True)
        self._launch(self.last_args)

    def _restore_settings(self, job):
        settings = read_json(job / "settings.json", {})
        if not isinstance(settings, dict):
            return
        for key, variable, options in (("duration", self.duration_value, DURATIONS), ("style", self.style_value, STYLES),
                                      ("look", self.look_value, LOOKS), ("quality", self.quality_value, QUALITIES)):
            match = next((label for label, value in options.items() if value == str(settings.get(key))), None)
            if match:
                variable.set(match)
        for key, variable in (("strength", self.strength_value), ("audio_level", self.audio_value)):
            try:
                variable.set(min(1, max(0, float(settings[key]))))
            except (KeyError, ValueError, TypeError):
                pass
        if settings.get("codec") in ("hevc", "h264"):
            self.codec_value.set("HEVC" if settings["codec"] == "hevc" else "H.264")
        paths = settings.get("inputs")
        if isinstance(paths, list):
            self.files, self.folder = [Path(path) for path in paths], None
            self._refresh_inputs()
        self.finish_value.set(FINISH_MODES[0])
        self.overrides_dirty = False

    def _load_recent_job(self):
        output = self.app_dir / "output"
        if not output.is_dir():
            return
        jobs = []
        for job in output.iterdir():
            status = read_json(job / "status.json", {}) if job.is_dir() else {}
            if isinstance(status, dict) and status.get("stage") == "complete" and (job / "preview.mp4").is_file():
                jobs.append(job)
        if not jobs:
            return
        self.job_dir = max(jobs, key=lambda job: (job / "status.json").stat().st_mtime_ns)
        self.last_args = ["make", "--job", str(self.job_dir)]
        self.terminal_stage = "complete"
        self._restore_settings(self.job_dir)
        self._load_candidates(force=True)
        self.progress["value"] = 100
        self.stage_text.set("Your latest session edit is ready — play the preview, review moments, or open its output folder.")
        self.detail_text.set(str(self.job_dir))
        self._refresh_outputs()

    def _render_final(self):
        if not self.job_dir:
            return
        args = self._without_options(self.last_args or ["make"], {"--job", "--preview-only", "--regenerate", "--overrides"})
        args += ["--job", str(self.job_dir)]
        args += ["--overrides", str(self._save_overrides())]
        self.last_args = args.copy()
        self._launch(args)

    def _regenerate(self):
        if not self.job_dir:
            return
        args = self._without_options(self._command_args(), {"--job", "--regenerate", "--overrides", "--input", "--folder"})
        args += ["--job", str(self.job_dir), "--regenerate", "--overrides", str(self._save_overrides())]
        self.last_args = args.copy()
        self.overrides_dirty = False
        self._launch(args)

    def _load_candidates(self, force=False):
        if not self.job_dir:
            return
        path = self.job_dir / "candidates.json"
        try:
            mtime = path.stat().st_mtime_ns
        except OSError:
            return
        if not force and mtime == self._candidate_mtime:
            return
        data = read_json(path, [])
        candidates = data.get("candidates", data.get("items", [])) if isinstance(data, dict) else data
        if not isinstance(candidates, list):
            self._log_warning("This candidate manifest has an unsupported format.")
            return
        self.candidates = [item for item in candidates if isinstance(item, dict) and item.get("id") is not None]
        timeline = read_json(self.job_dir / "timeline.json", {})
        ordered = {str(shot.get("id")): index for index, shot in enumerate(timeline.get("shots", []))} if isinstance(timeline, dict) else {}
        self.candidates.sort(key=lambda item: (not item.get("selected", False), ordered.get(str(item["id"]), len(ordered)), str(item.get("source", "")), float(item.get("start", 0))))
        self._candidate_mtime = mtime
        overrides = read_json(self.job_dir / "ui-overrides.json", None)
        if not isinstance(overrides, dict):
            overrides = read_json(self.job_dir / "overrides.json", {})
        self.override_choices = {key: [str(value) for value in overrides.get(key, [])] for key in ("keep", "exclude")}
        if isinstance(data, dict) and isinstance(data.get("overrides"), dict):
            self.overrides_dirty = any(set(self.override_choices[key]) != set(str(value) for value in data["overrides"].get(key, [])) for key in ("keep", "exclude"))
        self._paint_candidates()
        self.review_text.set(f"{len(self.candidates)} candidates; selected shots appear first. Scores are estimates. Double-click for exact source details.")

    def _paint_candidates(self):
        self.table.delete(*self.table.get_children())
        for index, item in enumerate(self.candidates):
            identifier = str(item["id"])
            choice = "Keep" if identifier in self.override_choices["keep"] else "Exclude" if identifier in self.override_choices["exclude"] else "Selected" if item.get("selected") else "Available"
            source = item.get("source", item.get("source_path", ""))
            reason = item.get("reason", item.get("reasons", ""))
            if isinstance(reason, list):
                reason = "; ".join(str(value) for value in reason)
            score = item.get("score", "")
            if isinstance(score, (int, float)):
                score = f"{score:.2f}"
            self.table.insert("", "end", iid=str(index), values=(choice, Path(str(source)).name, timestamp(item.get("start")), timestamp(item.get("end")), score, reason))
        self._refresh_outputs()

    def _show_candidate(self, _event=None):
        selected = self.table.selection()
        if not selected:
            return
        item = self.candidates[int(selected[0])]
        messagebox.showinfo("Source moment", json.dumps(item, indent=2, ensure_ascii=False), parent=self)

    def _set_choice(self, choice):
        selected = self.table.selection()
        if not selected:
            self.review_text.set("Select one or more moment rows first. Use Ctrl or Shift to select several.")
            return
        for index in selected:
            identifier = str(self.candidates[int(index)]["id"])
            for key in ("keep", "exclude"):
                self.override_choices[key] = [value for value in self.override_choices[key] if value != identifier]
            self.override_choices[choice].append(identifier)
        self.overrides_dirty = True
        self._save_overrides()
        self._paint_candidates()
        self.review_text.set("Choices saved. Click Regenerate edit to apply them to the proposed timeline.")

    def _reset_choices(self):
        self.override_choices = {"keep": [], "exclude": []}
        self.overrides_dirty = True
        self._save_overrides()
        self._paint_candidates()
        self.review_text.set("Overrides cleared. Regenerate edit to return to automatic selection.")

    def _save_overrides(self):
        if not self.job_dir:
            raise ValueError("No job has been created yet.")
        path = self.job_dir / "ui-overrides.json"
        write_json(path, self.override_choices)
        return path

    def _refresh_outputs(self):
        if not hasattr(self, "preview_button"):
            return
        busy = self.process is not None
        preview = bool(self.job_dir and (self.job_dir / "preview.mp4").is_file())
        current_pending = busy or self.overrides_dirty
        status = read_json(self.job_dir / "status.json", {}) if self.job_dir else {}
        completed_final = bool(self.job_dir and (self.job_dir / "final_4k.mp4").is_file()
                               and isinstance(status, dict) and status.get("stage") == "complete")
        self.preview_button.configure(state="normal" if preview else "disabled")
        self.play_final_button.configure(state="normal" if completed_final and not current_pending else "disabled")
        self.final_button.configure(state="normal" if preview and not current_pending else "disabled")
        editable = bool(self.job_dir and self.candidates and not busy)
        for button in (self.keep_button, self.exclude_button, self.reset_button, self.regenerate_button):
            button.configure(state="normal" if editable else "disabled")

    def _refresh_diagnostics(self):
        path = self.app_dir / "logs" / "diagnostics.json"
        try:
            mtime = path.stat().st_mtime_ns
        except OSError:
            return
        if mtime == self._last_diagnostics:
            return
        data = read_json(path, {})
        if not isinstance(data, dict):
            return
        self._last_diagnostics = mtime
        gpu = data.get("gpu_name") or "not detected / diagnostic details in run report"
        encoder = data.get("encoder") or "pending encode test"
        ai = bool(data.get("ai_available", False))
        self.gpu_text.set(f"GPU: {gpu}  •  Encoder: {encoder}  •  AI inference: {'tested and available' if ai else 'not available'}")
        self.quality_combo.configure(values=list(QUALITIES) if ai else ["Auto", "Lanczos"])
        if not ai and self.quality_value.get() == "AI (tested backend)":
            self.quality_value.set("Auto")
        self.quality_note.set("Auto: conventional GPU scaling when tested, with Lanczos fallback. AI is a separate, slower choice." if ai
                              else "AI enhancement unavailable. Auto uses conventional GPU scaling when tested, with Lanczos fallback.")
        warnings = data.get("warnings", [])
        for warning in warnings if isinstance(warnings, list) else [warnings]:
            self._log_warning(str(warning))

    def _open_path(self, path):
        try:
            if os.name == "nt":
                os.startfile(str(path))
            else:
                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])
        except OSError as exc:
            messagebox.showerror("Could not open file", str(exc), parent=self)

    def _play_preview(self):
        if self.job_dir and (self.job_dir / "preview.mp4").is_file():
            self._open_path(self.job_dir / "preview.mp4")

    def _play_final(self):
        self._refresh_outputs()
        if self.job_dir and self.play_final_button.instate(["!disabled"]):
            self._open_path(self.job_dir / "final_4k.mp4")

    def _open_output(self):
        path = self.job_dir or self.app_dir / "output"
        path.mkdir(parents=True, exist_ok=True)
        self._open_path(path)

    def _close(self):
        if self.process:
            if messagebox.askyesno("Render in progress", "Cancel at the next safe boundary and close? Completed work will be kept.", parent=self):
                self.closing = True
                self._cancel()
        else:
            self.destroy()

    def report_callback_exception(self, exc, value, tb):
        details = "".join(traceback.format_exception(exc, value, tb))
        self._log(details)
        log_path = self.app_dir / "logs" / "ui-error.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(details + "\n")
        messagebox.showerror("FPV Sesh", f"{value}\n\nDetails were saved in logs/ui-error.log.", parent=self)


def main():
    app = SeshApp()
    app.mainloop()


if __name__ == "__main__":
    main()
