"""Local desktop controls for the same CLI used by unattended FPV Sesh renders."""
from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import traceback
import uuid
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageDraw, ImageOps, ImageTk
from . import __version__


APP_DIR = Path(__file__).resolve().parents[1]
MEDIA_TYPES = [("Video recordings", "*.mp4 *.mov *.mkv *.m4v *.avi *.mts *.m2ts"), ("All files", "*.*")]
MUSIC_TYPES = [("Music and audio", "*.mp3 *.wav *.m4a *.aac *.flac *.ogg"), ("All files", "*.*")]
STYLES = {"Energetic highlights": "hype", "Cinematic": "cinematic", "Freestyle tricks": "freestyle", "Continuous flight": "flow"}
LOOKS = {"FPV Punch": "punch", "Natural": "natural", "Cinematic": "cinematic"}
QUALITIES = {"Auto": "auto", "Clean upscale": "lanczos", "AI detail (slower)": "ai"}
RECOGNITION_MODES = {"Automatic": "auto", "Off": "off", "Thorough": "thorough"}
FLIGHT_FILTERS = ("All motion", "Possible tricks", "Ordinary flight", "Uncertain")
DURATIONS = {"Auto": "auto", **{f"{n} seconds": str(n) for n in (15, 30, 60, 90, 120, 180)}}
FINISH_MODES = ("Render final automatically", "Stop at preview", "Wait for final approval")
EDIT_ORDERS = {"Story flow": "story", "Recording order": "chronological"}
MUSIC_ENDS = {"Fade out when the music ends": "fade", "Loop music to fill the edit": "loop"}
FRAMINGS = {"Full view with blurred background": "blur", "Full view with black bars": "fit", "Crop to fill the frame": "fill"}
SOCIAL_FORMATS = {
    "vertical": ("Vertical 9:16", "YouTube Shorts, Instagram Reels, TikTok, Facebook Reels, Snapchat"),
    "square": ("Square 1:1", "Square feed posts across social platforms"),
    "portrait": ("Portrait 4:5", "Portrait feed posts across social platforms"),
}
FLIGHT_LABELS = ("tree weave", "smooth line", "flip", "roll", "split-S", "powerloop", "dive", "landing", "crash", "other")
BG = "#111416"
SURFACE = "#1c2223"
FIELD = "#252d2e"
INK = "#f4f5ec"
MUTED = "#909996"
LIME = "#c5fa53"
LINE = "#343d3c"


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
        self.geometry("1280x800")
        self.minsize(1020, 650)
        self.configure(background=BG)
        self._thumb_requested: set[str] = set()
        self._thumb_images: dict[str, Image.Image] = {}
        self._thumb_errors: set[str] = set()
        self._clip_cards: list[dict] = []
        self._hero_key = None
        self._hero_source = None
        self._flight_data = {}
        self._flight_rows = []
        self._thumb_slots = threading.Semaphore(2)
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
        self.settings_dirty = False
        self.recognition_dirty = False
        self.mapping_only = False
        self._restoring_settings = False
        self.music_path: Path | None = None
        self._playback_paths: dict[str, Path] = {}
        self.run_preview_only = False
        self.terminal_stage = ""
        self._candidate_mtime = None
        self._flight_mtime = None
        self._last_diagnostics = None
        self.settings_widgets: list[tuple[tk.Widget, str]] = []
        self._make_styles()
        self._build()
        self._seed_inputs()
        self._refresh_optional_controls()
        self._refresh_diagnostics()
        self._load_recent_job()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.bind("<F1>", lambda _: self._show_help())
        self.bind("<FocusIn>", self._reveal_focus, add="+")
        self._poll_after = self.after(120, self._poll)

    def _make_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        self.option_add("*TCombobox*Listbox.background", FIELD)
        self.option_add("*TCombobox*Listbox.foreground", INK)
        self.option_add("*TCombobox*Listbox.selectBackground", LIME)
        self.option_add("*TCombobox*Listbox.selectForeground", BG)
        style.configure(".", background=SURFACE, foreground=INK, font=("Segoe UI", 10), bordercolor=LINE,
                        lightcolor=LINE, darkcolor=LINE, troughcolor=BG, focuscolor=LIME)
        style.configure("TFrame", background=SURFACE)
        style.configure("Workspace.TFrame", background=BG)
        style.configure("TLabel", background=SURFACE, foreground=INK)
        style.configure("Muted.TLabel", foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Outside.TLabel", background=BG, foreground=INK)
        style.configure("OutsideMuted.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Heading.TLabel", background=BG, foreground=INK, font=("Segoe UI Semibold", 26))
        style.configure("Eyebrow.TLabel", foreground=MUTED, font=("Segoe UI", 9, "bold"))
        style.configure("CardTitle.TLabel", font=("Segoe UI Semibold", 13))
        style.configure("TLabelframe", background=SURFACE, borderwidth=0, relief="flat")
        style.configure("TLabelframe.Label", background=SURFACE, foreground=INK, font=("Segoe UI Semibold", 11))
        style.configure("TButton", background=FIELD, foreground=INK, padding=(12, 9), borderwidth=0,
                        focusthickness=1, focuscolor=LIME, font=("Segoe UI Semibold", 9))
        style.map("TButton", background=[("disabled", "#1b2021"), ("active", "#36413c")],
                  foreground=[("disabled", "#69716f")])
        style.configure("Primary.TButton", background=LIME, foreground=BG, padding=(19, 12),
                        font=("Segoe UI", 11, "bold"))
        style.map("Primary.TButton", background=[("disabled", "#54653a"), ("active", "#d7ff81")],
                  foreground=[("disabled", "#929d86"), ("active", BG)])
        style.configure("TCheckbutton", background=SURFACE, foreground=INK, padding=(0, 4), indicatorbackground=FIELD,
                        indicatorforeground=BG, indicatormargin=(0, 0, 9, 0))
        style.map("TCheckbutton", background=[("active", SURFACE)], indicatorbackground=[("selected", LIME)],
                  foreground=[("disabled", MUTED)])
        for widget in ("TCombobox", "TSpinbox", "TEntry"):
            style.configure(widget, fieldbackground=FIELD, background=FIELD, foreground=INK, arrowcolor=LIME,
                            bordercolor=LINE, lightcolor=LINE, darkcolor=LINE, padding=(8, 7), selectbackground=LIME,
                            selectforeground=BG, insertcolor=INK)
            style.map(widget, fieldbackground=[("readonly", FIELD), ("disabled", "#1b2021")],
                      foreground=[("readonly", INK), ("disabled", MUTED)], background=[("active", "#35413a")],
                      bordercolor=[("focus", LIME)])
        slider = Image.new("RGBA", (16, 16))
        ImageDraw.Draw(slider).ellipse((1, 1, 14, 14), fill=LIME)
        self._slider_photo = ImageTk.PhotoImage(slider, master=self)
        style.element_create("Lime.Scale.slider", "image", self._slider_photo)
        style.layout("Horizontal.TScale", [("Horizontal.Scale.trough", {"sticky": "we", "children": [
            ("Lime.Scale.slider", {"side": "left", "sticky": ""})]})])
        style.configure("Horizontal.TScale", background=LIME, troughcolor=FIELD, bordercolor=FIELD,
                        lightcolor=FIELD, darkcolor=FIELD, borderwidth=0, gripcount=0)
        style.map("Horizontal.TScale", background=[("disabled", "#5c6950")])
        style.configure("Horizontal.TProgressbar", background=LIME, troughcolor=FIELD, borderwidth=0, thickness=4)
        style.configure("Vertical.TScrollbar", background="#46524b", troughcolor=BG, arrowcolor=MUTED, borderwidth=0)
        style.map("Vertical.TScrollbar", background=[("disabled", FIELD), ("active", "#617464"), ("!active", "#46524b")],
                  arrowcolor=[("disabled", "#536059"), ("!disabled", MUTED)],
                  lightcolor=[("!active", FIELD)], darkcolor=[("!active", FIELD)])
        style.configure("Treeview", background=SURFACE, fieldbackground=SURFACE, foreground=INK, borderwidth=0,
                        rowheight=34, font=("Segoe UI", 9))
        style.map("Treeview", background=[("selected", "#344522")], foreground=[("selected", LIME)])
        style.configure("Treeview.Heading", background=FIELD, foreground=MUTED, relief="flat",
                        padding=(8, 10), font=("Segoe UI", 9, "bold"))
        style.map("Treeview.Heading", background=[("active", "#334038")])
        style.configure("Studio.TNotebook", background=BG, borderwidth=0, tabmargins=0)
        style.layout("Studio.TNotebook.Tab", [])
        style.configure("Help.TNotebook", background=BG, borderwidth=0)
        style.configure("Help.TNotebook.Tab", background=FIELD, foreground=INK, padding=(10, 6))
        style.map("Help.TNotebook.Tab", background=[("selected", "#344522"), ("active", "#35413a")],
                  foreground=[("selected", LIME), ("active", INK)])
        self._ttk_style = style

    def _build(self):
        shell = ttk.Frame(self, style="Workspace.TFrame")
        shell.pack(fill="both", expand=True)
        self.sidebar = tk.Frame(shell, background=BG, width=174)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        brand = tk.Canvas(self.sidebar, width=146, height=108, background=BG, highlightthickness=0)
        brand.pack(padx=14, pady=(16, 16))
        brand.create_text(10, 14, anchor="nw", text="FPV", fill=INK, font=("Segoe UI", 28, "bold"))
        brand.create_text(10, 53, anchor="nw", text="SESH", fill=LIME, font=("Segoe UI", 28, "bold"))
        tk.Label(self.sidebar, text="CREATOR STUDIO", background=BG, foreground=MUTED,
                 font=("Segoe UI", 8, "bold"), anchor="w").pack(fill="x", padx=24, pady=(0, 14))
        self.nav_frame = tk.Frame(self.sidebar, background=BG)
        self.nav_frame.pack(fill="x", padx=12)
        self.workspace_text = tk.StringVar(value="No saved edit selected")
        tk.Label(self.sidebar, text="LOCAL WORKSPACE", background=BG, foreground="#626d67",
                 font=("Segoe UI", 8, "bold"), anchor="w").pack(side="bottom", fill="x", padx=24, pady=(0, 17))
        tk.Label(self.sidebar, textvariable=self.workspace_text, background=BG, foreground=MUTED,
                 font=("Segoe UI", 9), justify="left", anchor="w", wraplength=130).pack(side="bottom", fill="x", padx=24, pady=(0, 9))
        self.help_button = ttk.Button(self.sidebar, text="Help & setup", command=self._show_help)
        self.help_button.pack(side="bottom", fill="x", padx=12, pady=(0, 10))
        tk.Frame(shell, background="#252d2b", width=1).pack(side="left", fill="y")
        main = ttk.Frame(shell, style="Workspace.TFrame", padding=(22, 18, 20, 16))
        main.pack(side="left", fill="both", expand=True)

        header = ttk.Frame(main, style="Workspace.TFrame")
        header.pack(fill="x", pady=(0, 19))
        self.page_title = tk.StringVar(value="Session")
        self.page_description = tk.StringVar(value="Import footage, choose an edit, then render.")
        ttk.Label(header, textvariable=self.page_title, style="Heading.TLabel").pack(anchor="w")
        ttk.Label(header, textvariable=self.page_description, style="OutsideMuted.TLabel").pack(anchor="w", pady=(4, 0))
        self.gpu_status = tk.StringVar(value="Local processing")
        self.gpu_text = tk.StringVar(value="Hardware is checked when processing starts. Open Help & setup for installation and troubleshooting.")
        self.gpu_badge = tk.Label(header, textvariable=self.gpu_status, background="#26321e", foreground=LIME,
                                 font=("Segoe UI", 9, "bold"), padx=13, pady=8)
        self.gpu_badge.place(relx=1, x=-2, y=9, anchor="ne")

        dock = ttk.Frame(main, padding=(14, 11))
        dock.pack(side="bottom", fill="x", pady=(16, 0))
        self.stage_text = tk.StringVar(value="Ready when you are.")
        stage = ttk.Label(dock, textvariable=self.stage_text, font=("Segoe UI Semibold", 10), wraplength=960)
        stage.pack(anchor="w")
        self.progress = ttk.Progressbar(dock, maximum=100, mode="determinate")
        self.progress.pack(fill="x", pady=(8, 7))
        self.detail_text = tk.StringVar(value="Add your recordings to start. Original footage stays untouched.")
        detail = ttk.Label(dock, textvariable=self.detail_text, style="Muted.TLabel", wraplength=960)
        detail.pack(anchor="w")
        actions = ttk.Frame(dock)
        actions.pack(fill="x", pady=(10, 0))
        self.make_button = ttk.Button(actions, text="Make my sesh", style="Primary.TButton", command=self._start_new)
        self.make_button.pack(side="left")
        self.pause_button = ttk.Button(actions, text="Pause", command=self._pause, state="disabled")
        self.pause_button.pack(side="left", padx=(8, 4))
        self.cancel_button = ttk.Button(actions, text="Cancel", command=self._cancel, state="disabled")
        self.cancel_button.pack(side="left")
        self.resume_button = ttk.Button(actions, text="Resume…", command=self._resume_job)
        self.resume_button.pack(side="left", padx=4)
        self.play_final_button = ttk.Button(actions, text="Play 4K", command=self._play_final, state="disabled")
        self.play_final_button.pack(side="right")
        self.preview_button = ttk.Button(actions, text="Preview · 720p", command=self._play_preview, state="disabled")
        self.preview_button.pack(side="right", padx=(0, 6))
        ttk.Button(actions, text="Files", command=self._open_output).pack(side="right", padx=(0, 6))
        dock.bind("<Configure>", lambda event: [label.configure(wraplength=max(300, event.width - 32)) for label in (stage, detail)])

        self.notebook = ttk.Notebook(main, style="Studio.TNotebook")
        self.notebook.pack(fill="both", expand=True)
        self._pages = {}
        self._scroll_pages = {}
        session = self._new_page("Session")
        music = self._new_page("Music & sound")
        social = self._new_page("Social exports")
        review = self._new_page("Review moments", scroll=False)
        log = self._new_page("Progress & warnings", scroll=False)

        session.columnconfigure(0, weight=11, uniform="session")
        session.columnconfigure(1, weight=10, uniform="session")
        footage = ttk.Frame(session, padding=14)
        footage.grid(row=0, column=0, sticky="new", padx=(0, 12))
        ttk.Label(footage, text="YOUR FOOTAGE", style="Eyebrow.TLabel").pack(anchor="w", pady=(0, 10))
        self.hero_canvas = tk.Canvas(footage, height=220, background="#090d0b", highlightthickness=0)
        self.hero_canvas.pack(fill="x")
        self.hero_canvas.bind("<Configure>", lambda _: self._paint_hero())
        self.hero_canvas.bind("<Double-Button-1>", lambda _: self._open_hero())
        caption = ttk.Frame(footage)
        caption.pack(fill="x", pady=(9, 7))
        self.hero_title = tk.StringVar(value="Choose your first recording")
        self.hero_subtitle = tk.StringVar(value="A frame from your footage will appear here.")
        ttk.Label(caption, textvariable=self.hero_title, font=("Segoe UI Semibold", 11)).pack(anchor="w")
        ttk.Label(caption, textvariable=self.hero_subtitle, style="Muted.TLabel", wraplength=400).pack(anchor="w", pady=(3, 0))
        self.clip_strip = ttk.Frame(footage)
        self.clip_strip.pack(fill="x")
        self.clip_strip.bind("<Configure>", lambda _: self._paint_clip_cards())
        # Selection model remains a Listbox so keyboard/remove semantics stay consistent.
        self.input_list = tk.Listbox(footage, selectmode="extended", exportselection=False, background=FIELD,
                                    foreground=INK, selectbackground=LIME, selectforeground=BG)
        input_actions = ttk.Frame(footage)
        input_actions.pack(fill="x", pady=(11, 0))
        self.add_button = ttk.Button(input_actions, text="+ Add clips", command=self._choose_files)
        self.add_button.pack(side="left")
        self.folder_button = ttk.Button(input_actions, text="Folder", command=self._choose_folder)
        self.folder_button.pack(side="left", padx=5)
        self.remove_button = ttk.Button(input_actions, text="Remove", command=self._remove_files)
        self.remove_button.pack(side="left")
        self.input_text = tk.StringVar()
        ttk.Label(footage, textvariable=self.input_text, style="Muted.TLabel", wraplength=380).pack(anchor="w", pady=(12, 0))

        options = ttk.Frame(session, padding=14)
        options.grid(row=0, column=1, sticky="new")
        options._wide_combo = True
        options.columnconfigure(1, weight=1)
        options.columnconfigure(2, weight=1)
        ttk.Label(options, text="EDIT DIRECTION", style="Eyebrow.TLabel").grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 10))
        self.style_value = tk.StringVar(value="Energetic highlights")
        self.duration_value = tk.StringVar(value="Auto")
        self.look_value = tk.StringVar(value="Natural")
        self.quality_value = tk.StringVar(value="Auto")
        self.codec_value = tk.StringVar(value="HEVC")
        self.finish_value = tk.StringVar(value=FINISH_MODES[0])
        self.strength_value = tk.DoubleVar(value=0)
        self.audio_value = tk.DoubleVar(value=.4)
        self.order_value = tk.StringVar(value="Story flow")
        self.recovery_value = tk.StringVar(value="2.5")
        self.recognition_value = tk.StringVar(value="Automatic")
        presets = ttk.Frame(options)
        presets.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(0, 10))
        presets.columnconfigure((0, 1), weight=1, uniform="presets")
        self._preset_buttons = {}
        for index, (value, short, note) in enumerate((
                ("Energetic highlights", "Energetic", "Varied highlights"),
                ("Cinematic", "Cinematic", "Space to breathe"),
                ("Freestyle tricks", "Freestyle", "Trick-focused cuts"),
                ("Continuous flight", "Continuous", "Longer flight lines"))):
            button = tk.Button(presets, text=short + "\n" + note, anchor="w", justify="left", padx=12, pady=10,
                               relief="flat", borderwidth=0, highlightthickness=1, cursor="hand2",
                               background=FIELD, foreground=INK, activebackground="#344228", activeforeground=LIME,
                               font=("Segoe UI", 9), command=lambda choice=value: self.style_value.set(choice))
            button.grid(row=index // 2, column=index % 2, sticky="ew", padx=(0 if index % 2 == 0 else 7, 0), pady=(0, 7))
            self._preset_buttons[value] = button
            self.settings_widgets.append((button, "normal"))
        self._combo(options, 2, 0, "Duration", self.duration_value, list(DURATIONS), width=15)
        self._combo(options, 3, 0, "Color look", self.look_value, list(LOOKS), width=15)
        self._slider(options, 4, "Color strength", self.strength_value)
        self.quality_combo = self._combo(options, 5, 0, "Quality", self.quality_value, ["Auto", "Clean upscale"], width=15)
        self._combo(options, 6, 0, "Order", self.order_value, list(EDIT_ORDERS), width=15)
        self._combo(options, 7, 0, "Codec", self.codec_value, ["HEVC", "H.264"], width=15)
        self._combo(options, 8, 0, "After preview", self.finish_value, FINISH_MODES, width=25)
        ttk.Label(options, text="Recovery seconds").grid(row=9, column=0, sticky="w", pady=6)
        recovery = ttk.Spinbox(options, from_=2.5, to=5, increment=.5, textvariable=self.recovery_value, width=8)
        recovery.grid(row=9, column=1, columnspan=3, sticky="ew", pady=4)
        self.settings_widgets.append((recovery, "normal"))
        ttk.Label(options, text="Seconds of continued flight after estimated tricks.", style="Muted.TLabel",
                  wraplength=330).grid(row=10, column=0, columnspan=4, sticky="w", pady=(3, 9))
        self.recognition_combo = self._combo(options, 11, 0, "Flight recognition", self.recognition_value,
                                             list(RECOGNITION_MODES), width=15)
        self.recognition_note = ttk.Label(options, text="Runs locally using an internet-trained model. Trick labels are estimates.",
                                           style="Muted.TLabel", wraplength=330)
        self.recognition_note.grid(row=12, column=0, columnspan=4, sticky="w", pady=(3, 9))
        self.quality_note = tk.StringVar(value="Auto uses tested GPU scaling with a Lanczos fallback.")
        self.quality_note_label = ttk.Label(options, textvariable=self.quality_note, style="Muted.TLabel", wraplength=330)
        self.quality_note_label.grid(row=13, column=0, columnspan=4, sticky="w")
        options.bind("<Configure>", lambda event: [label.configure(wraplength=max(200, event.width - 30))
                                                    for label in (self.quality_note_label, self.recognition_note)])

        self.music_text = tk.StringVar(value="No music selected — source sound only.")
        track = ttk.LabelFrame(music, text="Choose a music file", padding=12)
        track.pack(fill="x")
        ttk.Label(track, textvariable=self.music_text, wraplength=1030).pack(anchor="w")
        track_actions = ttk.Frame(track)
        track_actions.pack(fill="x", pady=(9, 0))
        self.music_button = ttk.Button(track_actions, text="Choose music…", command=self._choose_music)
        self.music_button.pack(side="left")
        self.remove_music_button = ttk.Button(track_actions, text="Remove music", command=self._remove_music)
        self.remove_music_button.pack(side="left", padx=6)
        self.settings_widgets.extend(((self.music_button, "normal"), (self.remove_music_button, "normal")))
        ttk.Label(track_actions, text="MP3, WAV, M4A, AAC, FLAC or OGG", style="Muted.TLabel").pack(side="left", padx=10)
        mix = ttk.LabelFrame(music, text="Mix and timing", padding=12)
        mix.pack(fill="x", pady=(12, 0))
        mix.columnconfigure(1, weight=1)
        mix.columnconfigure(2, weight=1)
        self.music_level_value = tk.DoubleVar(value=.75)
        self.music_offset_value = tk.StringVar(value="0")
        self.music_fade_value = tk.StringVar(value="1.5")
        self.music_end_value = tk.StringVar(value=next(iter(MUSIC_ENDS)))
        self.beat_value = tk.BooleanVar(value=True)
        self._slider(mix, 0, "Music volume", self.music_level_value)
        self._slider(mix, 1, "Source sound", self.audio_value)
        timing = ttk.Frame(mix)
        timing.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(7, 0))
        for index, (label, variable, maximum, step) in enumerate((
                ("Start music at (seconds)", self.music_offset_value, 86400, 1),
                ("Fade length (seconds)", self.music_fade_value, 30, .5))):
            ttk.Label(timing, text=label).grid(row=0, column=index * 2, sticky="w", padx=(0 if not index else 24, 8))
            spin = ttk.Spinbox(timing, from_=0, to=maximum, increment=step, textvariable=variable, width=9)
            spin.grid(row=0, column=index * 2 + 1, sticky="w")
            self.settings_widgets.append((spin, "normal"))
        self._combo(mix, 3, 0, "If music is shorter", self.music_end_value, list(MUSIC_ENDS), width=34)
        beat = ttk.Checkbutton(mix, text="Favor music beats (respects recovery)", variable=self.beat_value)
        beat.grid(row=4, column=0, columnspan=4, sticky="w", pady=(8, 0))
        self.settings_widgets.append((beat, "normal"))
        ttk.Label(mix, text="Beat timing is used only where safe; it will not force every cut onto a beat.",
                  style="Muted.TLabel").grid(row=5, column=0, columnspan=4, sticky="w", pady=(4, 0))

        formats = ttk.LabelFrame(social, text="Export sizes", padding=12)
        formats.pack(fill="x")
        ttk.Label(formats, text="Landscape 4K master is always included.", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
        self.social_values = {}
        for row, (code, (label, explanation)) in enumerate(SOCIAL_FORMATS.items(), 1):
            variable = self.social_values[code] = tk.BooleanVar(value=False)
            check = ttk.Checkbutton(formats, text=label, variable=variable)
            check.grid(row=row, column=0, sticky="w", padx=(0, 24), pady=3)
            ttk.Label(formats, text=explanation, style="Muted.TLabel").grid(row=row, column=1, sticky="w")
            self.settings_widgets.append((check, "normal"))
        framing = ttk.LabelFrame(social, text="Keep the flight in view", padding=12)
        framing.pack(fill="x", pady=(10, 0))
        framing.columnconfigure(1, weight=1)
        framing.columnconfigure(2, weight=1)
        self.framing_value = tk.StringVar(value=next(iter(FRAMINGS)))
        self.focus_value = tk.DoubleVar(value=.5)
        self._combo(framing, 0, 0, "Framing", self.framing_value, list(FRAMINGS), width=39)
        self._slider(framing, 1, "Crop focus", self.focus_value)
        self.focus_scale = self.settings_widgets[-1][0]
        ttk.Label(framing, text="Focus moves a cropped view from left (0%) to right (100%). Full-view modes retain the entire picture.",
                  style="Muted.TLabel", wraplength=1000).grid(row=2, column=0, columnspan=4, sticky="w", pady=(4, 0))
        playback = ttk.Frame(social)
        playback.pack(fill="x", pady=(12, 0))
        self.social_play_value = tk.StringVar(value="")
        self.social_play_combo = ttk.Combobox(playback, textvariable=self.social_play_value, state="readonly", width=44)
        self.social_play_combo.pack(side="left", fill="x", expand=True)
        self.social_play_button = ttk.Button(playback, text="Play selected export", command=self._play_social, state="disabled")
        self.social_play_button.pack(side="left", padx=(8, 0))
        self.social_ready_text = tk.StringVar(value="Social previews and final exports appear here after they are rendered.")
        ttk.Label(social, textvariable=self.social_ready_text, style="Muted.TLabel").pack(anchor="w", pady=(3, 0))

        self.review_text = tk.StringVar(value="Select moments to keep or exclude, then regenerate your edit.")
        ttk.Label(review, textvariable=self.review_text, style="Muted.TLabel", wraplength=1100).pack(anchor="w", pady=(0, 7))
        teaching = ttk.Frame(review)
        teaching.pack(fill="x", pady=(0, 8))
        ttk.Label(teaching, text="Label selected moments").pack(side="left", padx=(0, 8))
        self.flight_label_value = tk.StringVar(value=FLIGHT_LABELS[0])
        label_combo = ttk.Combobox(teaching, textvariable=self.flight_label_value, values=FLIGHT_LABELS, state="readonly", width=16)
        label_combo.pack(side="left")
        self.settings_widgets.append((label_combo, "readonly"))
        self.teach_button = ttk.Button(teaching, text="Teach this moment", command=self._teach_moment, state="disabled")
        self.teach_button.pack(side="left", padx=8)
        ttk.Label(teaching, text="Optional: confirm or correct a moment.", style="Muted.TLabel").pack(side="left")
        review_actions = ttk.Frame(review)
        review_actions.pack(side="bottom", fill="x", pady=(8, 0))
        self.keep_button = ttk.Button(review_actions, text="Keep", command=lambda: self._set_choice("keep"), state="disabled")
        self.keep_button.pack(side="left")
        self.exclude_button = ttk.Button(review_actions, text="Exclude", command=lambda: self._set_choice("exclude"), state="disabled")
        self.exclude_button.pack(side="left", padx=4)
        self.reset_button = ttk.Button(review_actions, text="Clear overrides", command=self._reset_choices, state="disabled")
        self.reset_button.pack(side="left")
        self.source_button = ttk.Button(review_actions, text="Open source", command=self._open_selected_source, state="disabled")
        self.source_button.pack(side="left", padx=4)
        self.range_button = ttk.Button(review_actions, text="Add exact range…", command=self._add_exact_range, state="disabled")
        self.range_button.pack(side="left")
        self.regenerate_button = ttk.Button(review_actions, text="Regenerate edit", command=self._regenerate, state="disabled")
        self.regenerate_button.pack(side="right")
        self.final_button = ttk.Button(review_actions, text="Render final 4K", command=self._render_final, state="disabled")
        self.final_button.pack(side="right", padx=6)
        table_frame = ttk.Frame(review)
        table_frame.pack(fill="both", expand=True)
        columns = ("choice", "source", "start", "end", "score", "reason")
        self.table = ttk.Treeview(table_frame, columns=columns, show="headings", selectmode="extended", height=7)
        for col, title, width in (("choice", "Choice", 75), ("source", "Source recording", 130), ("start", "In", 70),
                                  ("end", "Out", 70), ("score", "Rank", 55), ("reason", "What happened / why selected", 460)):
            self.table.heading(col, text=title)
            self.table.column(col, width=width, minwidth=45, stretch=False)
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.table.pack(side="left", fill="both", expand=True)
        self.table.bind("<Configure>", lambda event: self.table.column("reason", width=max(150, event.width - 402)))
        self.table.bind("<Double-1>", self._show_candidate)
        self.table.bind("<Return>", self._show_candidate)
        hardware = ttk.Frame(log, padding=14)
        hardware.pack(fill="x", pady=(0, 12))
        ttk.Label(hardware, text="LOCAL RENDER ENGINE", style="Eyebrow.TLabel").pack(anchor="w", pady=(0, 7))
        hardware_label = ttk.Label(hardware, textvariable=self.gpu_text, wraplength=900)
        hardware_label.pack(anchor="w")
        ttk.Label(hardware, text="Gyro stabilization requires validated gyro data and lens calibration; it remains unavailable.",
                  style="Muted.TLabel", wraplength=900).pack(anchor="w", pady=(6, 0))
        hardware.bind("<Configure>", lambda event: hardware_label.configure(wraplength=max(300, event.width - 30)))
        self.log_box = tk.Text(log, height=12, wrap="word", state="disabled", font=("Consolas", 9), background=SURFACE, foreground=INK, insertbackground=INK, relief="flat", padx=14, pady=14)
        self.log_box.pack(fill="both", expand=True)

        flight = self._new_page("Flight map", position=4, scroll=False)
        self.learning_text = tk.StringVar(value="Online-trained video recognition and estimated flight events appear after analysis. Local examples are optional.")
        learning_label = ttk.Label(flight, textvariable=self.learning_text, style="OutsideMuted.TLabel", wraplength=960)
        learning_label.pack(anchor="w", pady=(0, 10))
        flight.bind("<Configure>", lambda event: learning_label.configure(wraplength=max(400, event.width - 12)))
        filters = ttk.Frame(flight, style="Workspace.TFrame")
        filters.pack(fill="x", pady=(0, 10))
        ttk.Label(filters, text="Show", style="OutsideMuted.TLabel").pack(side="left", padx=(0, 9))
        self.flight_filter_value = tk.StringVar(value="All motion")
        self.flight_filter_combo = ttk.Combobox(filters, textvariable=self.flight_filter_value,
                                                values=FLIGHT_FILTERS, state="readonly", width=22)
        self.flight_filter_combo.pack(side="left")
        self.map_button = ttk.Button(filters, text="Refresh understanding", command=self._refresh_understanding, state="disabled")
        self.map_button.pack(side="right")
        self.watch_section_button = ttk.Button(filters, text="Watch section", command=self._watch_flight_section, state="disabled")
        self.watch_section_button.pack(side="right", padx=(0, 8))
        self.flight_filter_value.trace_add("write", lambda *_: self._paint_flight_rows())
        self.flight_canvas = tk.Canvas(flight, height=166, background=SURFACE, highlightthickness=0)
        self.flight_canvas.pack(fill="x", pady=(0, 12))
        self.flight_canvas.bind("<Configure>", lambda _: self._paint_flight_timeline())
        flight_columns = ("source", "start", "end", "label", "confidence", "evidence")
        self.flight_table = ttk.Treeview(flight, columns=flight_columns, show="headings", height=8)
        for key, heading, width in (("source", "Recording", 140), ("start", "In", 70), ("end", "Out", 70),
                                    ("label", "What happened", 185), ("confidence", "Evidence", 90),
                                    ("evidence", "Why / method", 440)):
            self.flight_table.heading(key, text=heading)
            self.flight_table.column(key, width=width, minwidth=50, stretch=False)
        flight_scroll = ttk.Scrollbar(flight, orient="vertical", command=self.flight_table.yview)
        self.flight_table.configure(yscrollcommand=flight_scroll.set)
        flight_scroll.pack(side="right", fill="y")
        self.flight_table.pack(side="left", fill="both", expand=True)
        self.flight_table.bind("<Configure>", lambda event: self.flight_table.column("evidence", width=max(150, event.width - 557)))
        self.flight_table.bind("<Double-1>", self._show_flight_event)
        self.flight_table.bind("<Return>", self._show_flight_event)
        self.flight_table.bind("<<TreeviewSelect>>", lambda _: self._refresh_watch_section())

        variables = [self.style_value, self.duration_value, self.look_value, self.quality_value, self.codec_value,
                     self.strength_value, self.audio_value, self.order_value, self.recovery_value,
                     self.music_level_value, self.music_offset_value, self.music_fade_value, self.music_end_value,
                     self.beat_value, self.framing_value, self.focus_value, *self.social_values.values()]
        for variable in variables:
            variable.trace_add("write", self._settings_changed)
        self.recognition_value.trace_add("write", self._recognition_changed)
        self._build_navigation()
        self._polish_cards(self.notebook)
        self.bind("<Configure>", self._on_window_resize, add="+")
        self._refresh_dependent_controls()

    def _polish_cards(self, widget):
        for child in widget.winfo_children():
            self._polish_cards(child)
            if isinstance(child, ttk.LabelFrame):
                label = ttk.Label(child, text=child.cget("text"), style="CardTitle.TLabel", padding=(12, 7))
                child.configure(labelwidget=label)

    def _on_window_resize(self, event):
        if event.widget is self:
            height = min(220, max(125, event.height - 525))
            if int(self.hero_canvas.cget("height")) != height:
                self.hero_canvas.configure(height=height)

    def _reveal_focus(self, event):
        widget = event.widget
        page = self.notebook.select()
        canvas = self._scroll_pages.get(str(page))
        if not canvas or not str(widget).startswith(str(canvas) + "."):
            return
        self.update_idletasks()
        top = widget.winfo_rooty() - canvas.winfo_rooty()
        bottom = top + widget.winfo_height()
        region = canvas.bbox("all")
        if not region or region[3] <= canvas.winfo_height():
            return
        delta = top - 10 if top < 10 else bottom - canvas.winfo_height() + 10 if bottom > canvas.winfo_height() - 10 else 0
        if delta:
            canvas.yview_moveto((canvas.canvasy(0) + delta) / region[3])

    def _new_page(self, label, position=None, scroll=True):
        outer = ttk.Frame(self.notebook, style="Workspace.TFrame")
        if position is None:
            self.notebook.add(outer, text=label)
        else:
            self.notebook.insert(position, outer, text=label)
        self._pages[label] = outer
        if not scroll:
            return outer
        canvas = tk.Canvas(outer, background=BG, highlightthickness=0, borderwidth=0)
        bar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        bar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        content = ttk.Frame(canvas, style="Workspace.TFrame")
        window = canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.configure(yscrollcommand=bar.set)
        content.bind("<Configure>", lambda _: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=event.width))
        def wheel(event):
            if str(self.notebook.select()) == str(outer):
                canvas.yview_scroll(int(-event.delta / 120), "units")
        self.bind("<MouseWheel>", wheel, add="+")
        self._scroll_pages[str(outer)] = canvas
        return content

    @staticmethod
    def _round_rect(canvas, x0, y0, x1, y1, radius=12, **kwargs):
        points = [x0+radius,y0,x1-radius,y0,x1,y0,x1,y0+radius,x1,y1-radius,x1,y1,
                  x1-radius,y1,x0+radius,y1,x0,y1,x0,y1-radius,x0,y0+radius,x0,y0]
        return canvas.create_polygon(points, smooth=True, splinesteps=18, **kwargs)

    def _build_navigation(self):
        self._nav_items = {}
        short_names = {"Session": "Session", "Music & sound": "Music", "Social exports": "Social",
                       "Review moments": "Moments", "Flight map": "Flight map", "Progress & warnings": "Activity"}
        for index, tab in enumerate(self.notebook.tabs()):
            label = self.notebook.tab(tab, "text")
            tile = tk.Canvas(self.nav_frame, width=150, height=46, background=BG, highlightthickness=0,
                             cursor="hand2", takefocus=True)
            tile.pack(fill="x", pady=3)
            tile.bind("<Button-1>", lambda _, page=tab: self.notebook.select(page))
            tile.bind("<Return>", lambda _, page=tab: self.notebook.select(page))
            tile.bind("<space>", lambda _, page=tab: self.notebook.select(page))
            tile.bind("<FocusIn>", lambda _: self._sync_navigation())
            tile.bind("<FocusOut>", lambda _: self._sync_navigation())
            self._nav_items[tab] = (tile, short_names[label], index + 1)
        self.notebook.bind("<<NotebookTabChanged>>", lambda _: self._sync_navigation())
        self._sync_navigation()

    def _sync_navigation(self):
        if not hasattr(self, "_nav_items"):
            return
        selected = self.notebook.select()
        descriptions = {
            "Session": "Select your recordings and shape the edit.",
            "Music & sound": "Set the soundtrack, mix and timing.",
            "Social exports": "Keep the whole flight in view, in every format.",
            "Review moments": "Keep complete tricks. Give each line room to finish.",
            "Flight map": "Explore suggested tricks and flight lines along each recording.",
            "Progress & warnings": "Rendering progress, hardware and diagnostic details.",
        }
        for tab, (tile, label, number) in self._nav_items.items():
            tile.delete("all")
            active = tab == selected
            if active:
                self._round_rect(tile, 0, 1, 150, 44, fill="#2a3522", outline="")
                tile.create_oval(12, 20, 18, 26, fill=LIME, outline="")
            else:
                tile.create_text(15, 23, text=f"{number:02}", font=("Segoe UI", 8), fill="#5c6860")
            tile.create_text(31, 23, text=label, anchor="w", fill=LIME if active else MUTED,
                             font=("Segoe UI", 10, "bold" if active else "normal"))
            if self.focus_get() is tile:
                tile.create_rectangle(2, 2, 148, 43, outline=LIME, dash=(2, 2))
        if selected:
            full = self.notebook.tab(selected, "text")
            self.page_title.set("Activity" if full == "Progress & warnings" else full)
            self.page_description.set(descriptions.get(full, ""))

    def _sync_presets(self):
        if not hasattr(self, "_preset_buttons"):
            return
        for value, button in self._preset_buttons.items():
            selected = value == self.style_value.get()
            button.configure(background="#2a3522" if selected else FIELD, foreground=LIME if selected else INK,
                             highlightbackground=LIME if selected else LINE, highlightcolor=LIME)

    def _find_thumbnail_ffmpeg(self):
        from .media import locate_tools
        try:
            ffmpeg, _ = locate_tools()
            return Path(ffmpeg)
        except (OSError, RuntimeError, ValueError):
            return None

    def _request_thumbnail(self, source, seconds=12):
        source = Path(source)
        try:
            stat = source.stat()
        except OSError:
            return None
        signature = f"{source.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{seconds:.3f}"
        key = hashlib.sha256(signature.encode()).hexdigest()[:24]
        if key in self._thumb_requested:
            return key
        self._thumb_requested.add(key)
        output = self.app_dir / "cache" / "ui-thumbnails" / (key + ".jpg")
        def extract():
            try:
                if not output.is_file():
                    ffmpeg = self._find_thumbnail_ffmpeg()
                    if not ffmpeg:
                        self.events.put(("thumbnail", {"key": key, "error": "Frame extraction unavailable"}))
                        return
                    with self._thumb_slots:
                        if self.closing:
                            return
                        output.parent.mkdir(parents=True, exist_ok=True)
                        temporary = output.with_name(output.stem + ".tmp.jpg")
                        command = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-threads", "1",
                                   "-ss", f"{max(0, seconds):.3f}", "-i", str(source), "-frames:v", "1",
                                   "-vf", "scale=960:-2", "-q:v", "3", "-threads", "1", "-y", str(temporary)]
                        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                                timeout=30, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                        if result.returncode or not temporary.is_file():
                            temporary.unlink(missing_ok=True)
                            self.events.put(("thumbnail", {"key": key, "error": "Frame unavailable"}))
                            return
                        temporary.replace(output)
                self.events.put(("thumbnail", {"key": key, "path": str(output)}))
            except (OSError, subprocess.TimeoutExpired):
                self.events.put(("thumbnail", {"key": key, "error": "Frame unavailable"}))
                return
        threading.Thread(target=extract, daemon=True).start()
        return key

    def _on_thumbnail(self, data):
        if data.get("error"):
            self._thumb_errors.add(data["key"])
            self._paint_clip_cards()
            self._paint_hero()
            return
        try:
            with Image.open(data["path"]) as source:
                self._thumb_images[data["key"]] = source.convert("RGB").copy()
        except (OSError, ValueError):
            return
        if len(self._thumb_images) > 36:
            first = next(iter(self._thumb_images))
            if first != self._hero_key:
                self._thumb_images.pop(first, None)
        self._paint_clip_cards()
        self._paint_hero()

    def _select_clip(self, index, event=None):
        if not 0 <= index < len(self._clip_cards):
            return
        if not event or not event.state & 0x4:
            self.input_list.selection_clear(0, "end")
        self.input_list.selection_set(index)
        card = self._clip_cards[index]
        self._hero_source = card["source"]
        self._hero_key = card["key"]
        self.hero_title.set(card["source"].name)
        self.hero_subtitle.set("Source frame · double-click the image to open the recording")
        self._paint_clip_cards()
        self._paint_hero()

    def _paint_clip_cards(self):
        for index, card in enumerate(self._clip_cards):
            canvas = card["canvas"]
            if not canvas.winfo_exists():
                continue
            width = max(80, canvas.winfo_width())
            canvas.delete("all")
            selected = index in self.input_list.curselection()
            canvas.create_rectangle(1, 1, width-1, 98, fill=FIELD, outline=LIME if selected else LINE, width=1)
            picture = self._thumb_images.get(card["key"])
            if picture is not None:
                thumb = ImageOps.fit(picture, (max(1, width-8), 63), method=Image.Resampling.LANCZOS)
                card["photo"] = ImageTk.PhotoImage(thumb, master=self)
                canvas.create_image(4, 4, image=card["photo"], anchor="nw")
            else:
                status = "Frame unavailable" if card["key"] in self._thumb_errors else "Loading frame" if card["key"] else "Recording"
                canvas.create_text(width/2, 33, text=status, fill=MUTED, font=("Segoe UI", 8))
            canvas.create_text(8, 81, text=card["source"].name, fill=LIME if selected else INK,
                               font=("Segoe UI", 8), anchor="w", width=width-14)

    def _paint_hero(self):
        if not hasattr(self, "hero_canvas"):
            return
        canvas = self.hero_canvas
        width, height = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
        canvas.delete("all")
        picture = self._thumb_images.get(self._hero_key)
        if picture is not None and width > 1:
            fitted = ImageOps.contain(picture, (width, height), method=Image.Resampling.LANCZOS)
            self._hero_photo = ImageTk.PhotoImage(fitted, master=self)
            canvas.create_image(width/2, height/2, image=self._hero_photo)
            canvas.create_rectangle(12, 12, 122, 37, fill="#131a16", outline="")
            canvas.create_text(23, 25, text="SOURCE FRAME", fill=LIME, anchor="w", font=("Segoe UI", 8, "bold"))
        else:
            canvas.create_line(width/2-20, height/2, width/2+20, height/2, fill="#3c493e", width=1)
            canvas.create_line(width/2, height/2-20, width/2, height/2+20, fill="#3c493e", width=1)
            status = "Preview frame unavailable" if self._hero_key in self._thumb_errors else "Loading your footage…" if self._hero_key else "Add your recordings"
            canvas.create_text(width/2, height/2+47, text=status,
                               fill=MUTED, font=("Segoe UI", 10))

    def _open_hero(self):
        if self._hero_source and self._hero_source.is_file():
            self._open_path(self._hero_source)

    def _load_job_poster(self):
        if not self.job_dir:
            return
        timeline = read_json(self.job_dir / "timeline.json", {})
        shots = timeline.get("shots", []) if isinstance(timeline, dict) else []
        if shots:
            first = shots[0]
            source = Path(first.get("source", ""))
            key = self._request_thumbnail(source, float(first.get("start", 0)) + .5)
            if key:
                self._hero_key, self._hero_source = key, source
                self.hero_title.set(source.name)
                self.hero_subtitle.set(f"Opening shot · source {timestamp(float(first.get('start', 0)) + .5)}")
                for index, card in enumerate(self._clip_cards):
                    if card["source"] == source:
                        self.input_list.selection_clear(0, "end")
                        self.input_list.selection_set(index)
                        self._paint_clip_cards()
                        break
                self._paint_hero()
        self.workspace_text.set(self.job_dir.name)

    def _combo(self, parent, row, column, label, variable, values, width=19):
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", padx=(0 if column == 0 else 12, 7), pady=4)
        widget = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly", width=width)
        widget.grid(row=row, column=column + 1, columnspan=3 if getattr(parent, "_wide_combo", False) else 1, sticky="ew", pady=4)
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
        self.files, self.folder = [], None
        self._refresh_inputs()
        self.stage_text.set("Add recordings to start your first edit.")
        self.detail_text.set("Choose clips, optionally add music, then review a preview. Help & setup explains the next steps.")

    def _refresh_inputs(self):
        self.input_list.delete(0, "end")
        for child in self.clip_strip.winfo_children():
            child.destroy()
        self._clip_cards = []
        if self.folder:
            self.input_list.insert("end", f"Folder: {self.folder}")
            self.input_text.set("Folder selected · recordings will be scanned when rendering starts.")
            self.hero_title.set(self.folder.name)
            self.hero_subtitle.set("Session folder")
            self._hero_key = None
            self._hero_source = None
            self._paint_hero()
            self._refresh_outputs()
            return
        probes = read_json(self.job_dir / "sources.json", []) if self.job_dir else []
        durations = {item.get("source"): item.get("duration", 0) for item in probes if isinstance(item, dict)}
        timeline = read_json(self.job_dir / "timeline.json", {}) if self.job_dir else {}
        shots = timeline.get("shots", []) if isinstance(timeline, dict) else []
        for col in range(3):
            self.clip_strip.columnconfigure(col, weight=1, uniform="clips")
        for index, path in enumerate(self.files):
            self.input_list.insert("end", str(path))
            shot = next((item for item in shots if item.get("source") == str(path)), None)
            duration = durations.get(str(path), 0)
            seconds = float(shot.get("start", 0)) + .5 if shot else min(12, duration * .25) if duration else 1
            key = self._request_thumbnail(path, seconds)
            card = tk.Canvas(self.clip_strip, width=110, height=100, background=SURFACE, highlightthickness=1,
                             highlightbackground=SURFACE, highlightcolor=LIME, cursor="hand2", takefocus=True)
            card.grid(row=index // 3, column=index % 3, sticky="ew", padx=(0 if index % 3 == 0 else 5, 0), pady=(0, 5))
            card.bind("<Button-1>", lambda event, number=index: self._select_clip(number, event))
            card.bind("<Return>", lambda event, number=index: self._select_clip(number, event))
            card.bind("<space>", lambda event, number=index: self._select_clip(number, event))
            card.bind("<Delete>", lambda event, number=index: self._remove_focused_clip(number))
            card.bind("<Configure>", lambda _: self._paint_clip_cards())
            self._clip_cards.append({"canvas": card, "source": path, "key": key})
        self.input_text.set(f"{len(self.files)} recording{'s' if len(self.files) != 1 else ''} imported · Ctrl-click to select several." if self.files
                            else "Add clips or choose a folder. Original files stay untouched.")
        if self._clip_cards:
            self._select_clip(0)
        else:
            self.hero_title.set("Choose your first recording")
            self.hero_subtitle.set("A frame from your footage will appear here.")
            self._hero_key = self._hero_source = None
            self._paint_hero()
        self._refresh_outputs()

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
        if self.folder:
            self.folder = None
        else:
            self.files = [path for index, path in enumerate(self.files) if index not in selected]
        self._refresh_inputs()

    def _remove_focused_clip(self, index):
        if self.process is None:
            if index not in self.input_list.curselection():
                self._select_clip(index)
            self._remove_files()
        return "break"

    def _settings_changed(self, *_):
        if self._restoring_settings:
            return
        if self.job_dir:
            self.settings_dirty = True
            self.review_text.set("Settings changed. Regenerate the edit to apply them to this saved job.")
        self._refresh_dependent_controls()
        self._refresh_outputs()

    def _recognition_changed(self, *_):
        if self._restoring_settings:
            return
        if self.job_dir:
            self.recognition_dirty = True
            self.review_text.set("Recognition changed. Use Refresh understanding on the Flight map to update estimates without rendering.")
        self._refresh_optional_controls()
        self._refresh_outputs()

    def _refresh_dependent_controls(self):
        self._sync_presets()
        if hasattr(self, "focus_scale"):
            self.focus_scale.configure(state="normal" if self.process is None and FRAMINGS.get(self.framing_value.get()) == "fill" else "disabled")
        if hasattr(self, "remove_music_button"):
            self.remove_music_button.configure(state="normal" if self.music_path and self.process is None else "disabled")
        if hasattr(self, "music_text"):
            self.music_text.set(str(self.music_path) if self.music_path else "No music selected — source sound only.")

    def _optional_files_present(self):
        folder = self.app_dir / "models" / "qwen3-vl-2b"
        manifest = read_json(folder / "manifest.json", {})
        assets = manifest.get("assets", []) if isinstance(manifest, dict) else []
        if not isinstance(assets, list) or not assets:
            return False
        try:
            runtime = self.app_dir / ".venv-ai"
            if not (runtime / "Scripts/python.exe").is_file() or not (runtime / "Lib/site-packages/transformers/models/qwen3_vl").is_dir():
                return False
            for item in assets:
                path = (folder / item["file"]).resolve()
                if not path.is_relative_to(folder.resolve()) or not path.is_file() or path.stat().st_size != item["size_bytes"]:
                    return False
            return True
        except (OSError, TypeError, KeyError):
            return False

    def _refresh_optional_controls(self):
        """Fast presence hints only; the processing backend verifies integrity."""
        present = self._optional_files_present()
        if self.recognition_value.get() == "Off":
            note = "Video recognition is off. Motion estimates still work."
        else:
            note = "Runs locally using an internet-trained model. Trick labels are estimates. "
            note += ("Model files found; checked before analysis." if present else
                     "Optional model files are missing or incomplete; motion estimates still work. See Help & setup.")
        self.recognition_note.configure(text=note)
        data = read_json(self.app_dir / "logs/diagnostics.json", {})
        tested = isinstance(data, dict) and data.get("ai_available") is True
        ai_files = ((self.app_dir / ".venv-ai/Scripts/python.exe").is_file() and
                    (self.app_dir / "models/real-esrgan-cuda/RealESRGAN_x2plus.pth").is_file())
        available = tested and ai_files
        self.quality_combo.configure(values=list(QUALITIES) if available else ["Auto", "Clean upscale"])
        self.quality_note.set("AI detail has a saved local check and is verified again before rendering; it may change fine textures."
                              if available else "Auto and Clean upscale work without optional AI. AI detail needs installation and a local sample check; see Help & setup.")
        return {"video_files_present": present, "ai_saved_check": available}

    def _help_topics(self):
        status = self._refresh_optional_controls()
        return {
            "Get started": (
                "Install once with install.cmd, then open the studio with launch.cmd. Core editing supports 64-bit Python 3.12 or 3.13 with Tk; maintained Python 3.13 is recommended.\n\n"
                "1. Add clips or choose a folder of original recordings. Nothing is imported automatically.\n\n"
                "2. Choose the edit style and length. Natural color at 0% keeps the default look restrained. "
                "For a first run, choose Stop at preview under After preview.\n\n"
                "3. Music is optional. Add your own audio under Music, adjust its start and sound levels, or leave it empty for flight sound.\n\n"
                "4. Review Moments: Keep, Exclude, or Add exact range, then Regenerate edit. Press Enter on a selected row for full evidence. "
                "Flight map separates motion estimates from video observations; Watch section opens the original with context.\n\n"
                "5. Choose social shapes if wanted. Full view preserves the scene; Crop to fill removes its edges. "
                "Render final 4K when ready. Files opens the saved job.\n\n"
                "Changing imported clips starts a new session with Make my sesh. Regeneration uses the saved job's recordings. "
                "Original recordings are never replaced."),
            "Optional features": (
                f"Video model files: {'found (integrity is checked before analysis)' if status['video_files_present'] else 'missing or incomplete'}.\n"
                f"AI detail: {'saved local check available' if status['ai_saved_check'] else 'installation and a local sample check required'}.\n\n"
                "Ordinary editing works without either optional feature. Auto/Clean upscale are available. "
                "Automatic video recognition falls back to motion estimates when its optional model is unavailable; Off skips video recognition. "
                "Thorough takes longer. User examples are optional.\n\n"
                "The setup guide explains setup-ai.ps1, setup-video.ps1 and the optional setup-vision.ps1 scene model. "
                "Downloads can be several GB. The current optional AI packages require a separately supported runtime; check the setup guide before installing. "
                "AI detail also requires the validate-ai sample check; installation alone does not enable it.\n\n"
                "Recognition uses internet-trained weights locally. It does not upload your recordings or require an account/API key. "
                "Named tricks remain estimates. Measured image rotation and the video model's original interpretation stay separate."),
            "Troubleshooting": (
                "Cannot start: run install.cmd, then launch.cmd. Run doctor.cmd for a local readiness report; setup.ps1 -CheckOnly checks installation files. "
                "Core editing supports 64-bit Python 3.12 or 3.13 with Tk. Open the setup guide for optional AI runtime requirements.\n\n"
                "Processing stopped: open Activity for the underlying message. Check available disk space and that originals/music still exist. "
                "Resume selects a saved job; completed cached work is reused when valid.\n\n"
                "Pause/cancel: a request may wait for the current supported stage to reach a safe boundary. Cancel keeps completed work. "
                "A partial flight scan keeps finished observations; Refresh understanding continues it without rendering.\n\n"
                "Model unavailable: base editing still works. Install the optional components, then use Refresh availability here. "
                "The next analysis/render performs full checks. A prior diagnostic is not a promise that a changed GPU or model will work.\n\n"
                "Keyboard: Tab/Shift+Tab moves through controls and scrolls them into view. Focus a recording card and press Enter/Space to select; "
                "Ctrl+Space adds to the selection, Delete removes selected clips. Enter on a Moments/Flight map row opens details. F1 opens this help.\n\n"
                "Open logs for local diagnostic files. Logs can include local paths and filenames; inspect them before sharing."),
            "About": (
                f"FPV Sesh {__version__}\n\nA local Windows FPV editing studio.\n\n"
                "Original footage, music, jobs and exports stay on your computer. Optional model/software downloads have their own licenses. "
                "Read the repository LICENSE and THIRD_PARTY_NOTICES files for details.\n\n"
                "A 4K export does not create native camera detail. AI enhancement can alter texture; inspect a sample. "
                "Flight understanding is an estimate, not proof of a named trick, complete airborne recovery or a geographic route.")}

    def _show_help(self):
        previous = getattr(self, "_help_window", None)
        if previous is not None and previous.winfo_exists():
            previous.lift()
            return
        window = self._help_window = tk.Toplevel(self)
        window.withdraw()
        window.title("FPV Sesh — Help & setup")
        window.geometry("780x580")
        window.minsize(600, 440)
        window.configure(background=BG)
        book = ttk.Notebook(window, style="Help.TNotebook")
        self._help_texts = {}
        for title, body in self._help_topics().items():
            page = ttk.Frame(book)
            book.add(page, text=title)
            text = tk.Text(page, wrap="word", background=SURFACE, foreground=INK, font=("Segoe UI", 10),
                           padx=16, pady=16, relief="flat", insertbackground=INK)
            bar = ttk.Scrollbar(page, command=text.yview)
            text.configure(yscrollcommand=bar.set)
            bar.pack(side="right", fill="y")
            text.pack(fill="both", expand=True)
            text.insert("1.0", body)
            text.configure(state="disabled")
            text.bind("<Tab>", lambda event: (event.widget.tk_focusNext().focus_set(), "break")[1])
            text.bind("<Shift-Tab>", lambda event: (event.widget.tk_focusPrev().focus_set(), "break")[1])
            self._help_texts[title] = text
        actions = ttk.Frame(window)
        actions.pack(side="bottom", fill="x", padx=14, pady=(0, 14))
        ttk.Button(actions, text="Setup guide", command=lambda: self._open_path(self.app_dir / "README.md")).pack(side="left")
        ttk.Button(actions, text="Open logs", command=self._open_logs).pack(side="left", padx=7)
        ttk.Button(actions, text="Refresh availability", command=self._refresh_help).pack(side="left")
        ttk.Button(actions, text="Close", command=window.destroy).pack(side="right")
        book.pack(fill="both", expand=True, padx=14, pady=14)
        window.bind("<Escape>", lambda _: window.destroy())
        if float(self.attributes("-alpha")) == 0:
            window.attributes("-alpha", 0)
        window.deiconify()

    def _refresh_help(self):
        for title, body in self._help_topics().items():
            text = self._help_texts[title]
            text.configure(state="normal")
            text.delete("1.0", "end")
            text.insert("1.0", body)
            text.configure(state="disabled")

    def _open_logs(self):
        folder = self.app_dir / "logs"
        folder.mkdir(parents=True, exist_ok=True)
        self._open_path(folder)

    def _choose_music(self):
        selected = filedialog.askopenfilename(parent=self, title="Choose music for your session", filetypes=MUSIC_TYPES)
        if selected:
            self.music_path = Path(selected)
            self._settings_changed()

    def _remove_music(self):
        self.music_path = None
        self._settings_changed()

    @staticmethod
    def _number(variable, label, minimum, maximum):
        try:
            value = float(variable.get())
        except (ValueError, TypeError, tk.TclError):
            raise ValueError(f"{label} must be a number.") from None
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"{label} must be between {minimum:g} and {maximum:g}.")
        return value

    def _settings_args(self) -> list[str]:
        if self.music_path and not self.music_path.is_file():
            raise ValueError("The selected music file is missing. Choose it again or remove music.")
        args = [
            "--duration", DURATIONS[self.duration_value.get()], "--style", STYLES[self.style_value.get()],
            "--look", LOOKS[self.look_value.get()], "--strength", f"{self._number(self.strength_value, 'Color strength', 0, 1):.4f}",
            "--quality", QUALITIES[self.quality_value.get()], "--audio-level", f"{self._number(self.audio_value, 'Source volume', 0, 1):.4f}",
            "--codec", "hevc" if self.codec_value.get() == "HEVC" else "h264",
            "--edit-order", EDIT_ORDERS[self.order_value.get()],
            "--recognition", RECOGNITION_MODES[self.recognition_value.get()],
            "--recovery", f"{self._number(self.recovery_value, 'Recovery time', .5, 8):g}",
            "--music-level", f"{self._number(self.music_level_value, 'Music volume', 0, 1):.4f}",
            "--music-offset", f"{self._number(self.music_offset_value, 'Music start', 0, 86400):g}",
            "--music-fade", f"{self._number(self.music_fade_value, 'Music fade', 0, 30):g}",
            "--music-end", MUSIC_ENDS[self.music_end_value.get()],
            "--beat-sync" if self.beat_value.get() else "--no-beat-sync",
            "--social-formats", ",".join(code for code, variable in self.social_values.items() if variable.get()) or "none",
            "--framing", FRAMINGS[self.framing_value.get()],
            "--focus-x", f"{self._number(self.focus_value, 'Crop focus', 0, 1):.4f}",
        ]
        args += ["--music", str(self.music_path)] if self.music_path else ["--no-music"]
        return args

    def _command_args(self) -> list[str]:
        args = ["make"]
        if self.folder:
            args += ["--folder", str(self.folder)]
        else:
            for path in self.files:
                args += ["--input", str(path)]
        args += self._settings_args()
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
                index += 1 if item in ("--preview-only", "--regenerate", "--no-music", "--beat-sync", "--no-beat-sync") else 2
            else:
                cleaned.append(item)
                index += 1
        return cleaned

    def _start_new(self):
        if not self.folder and not self.files:
            messagebox.showinfo("Choose recordings", "Add one or more recordings or choose your session folder.", parent=self)
            return
        try:
            args = self._command_args()
        except ValueError as exc:
            messagebox.showinfo("Check your settings", str(exc), parent=self)
            return
        self.job_dir = None
        self._candidate_mtime = None
        self.candidates = []
        self.override_choices = {"keep": [], "exclude": []}
        self.overrides_dirty = False
        self.settings_dirty = False
        self.recognition_dirty = False
        self._paint_candidates()
        self.last_args = args
        self._launch(self.last_args)

    def _launch(self, args):
        if self.process and self.process.poll() is None:
            return
        self.paused = False
        self.mapping_only = bool(args and args[0] == "map-flight")
        self.run_preview_only = "--preview-only" in args
        self.terminal_stage = ""
        self.pause_button.configure(text="Pause")
        self.progress["value"] = 0
        self.stage_text.set("Refreshing flight understanding…" if self.mapping_only else "Starting the local renderer…")
        self.detail_text.set("Analyzing the saved recordings. Your finished edit stays unchanged." if self.mapping_only
                             else "Preparing this job. Original footage stays untouched.")
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
            self.stage_text.set("Flight analysis could not start." if self.mapping_only else "The renderer could not start.")
            self._log(str(exc))
            messagebox.showerror("Could not start FPV Sesh", str(exc), parent=self)
            return
        self._set_busy(True)
        self._log("Started flight analysis without rendering." if self.mapping_only else "Started a local render job.")
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
                elif kind == "thumbnail":
                    self._on_thumbnail(payload)
                elif kind == "exit":
                    self.process = None
                    self._set_busy(False)
                    self._load_candidates(force=True)
                    self._load_flight_map(force=True)
                    if not self.mapping_only:
                        self._load_job_poster()
                    self._refresh_outputs()
                    if self.mapping_only:
                        if self.terminal_stage == "cancelled":
                            self.stage_text.set("Flight analysis cancelled safely. Refresh understanding to continue; your finished edit is unchanged.")
                        elif payload == 0 and self.terminal_stage == "partial":
                            self.progress["value"] = 100
                            self.recognition_dirty = False
                            self.stage_text.set("Flight understanding partly updated. Completed observations are available.")
                            self.detail_text.set("Refresh understanding to continue. Your finished edit is unchanged.")
                        elif payload == 0 and self.terminal_stage != "error":
                            self.progress["value"] = 100
                            self.recognition_dirty = False
                            self.stage_text.set("Flight understanding updated. Your finished edit is unchanged.")
                        else:
                            self.stage_text.set("Flight analysis stopped. Your finished edit is unchanged; check the messages before trying again.")
                    elif self.terminal_stage == "cancelled":
                        self.stage_text.set("Cancelled safely — completed work is kept. Resume the saved job when ready.")
                    elif payload == 0:
                        self.progress["value"] = 100
                        self.recognition_dirty = False
                        if not self.run_preview_only and self.job_dir and (self.job_dir / "final_4k.mp4").is_file():
                            self.stage_text.set("Your session edit is ready — final 4K video and preview are saved.")
                        elif self.job_dir and (self.job_dir / "preview.mp4").is_file():
                            self.stage_text.set("Preview ready — review your moments or choose Render final 4K.")
                        else:
                            self.stage_text.set("Job finished. Check the run report and messages for its result.")
                    else:
                        self.stage_text.set("Job stopped — completed work is kept. Read the messages or resume the saved job.")
                    self._log(f"{'Flight analysis' if self.mapping_only else 'Renderer'} exited with code {payload}.")
                    if self.closing:
                        self.destroy()
                        return
        except queue.Empty:
            pass
        self._load_candidates()
        self._load_flight_map()
        self._refresh_diagnostics()
        self._refresh_outputs()
        self._refresh_dependent_controls()
        self._poll_after = self.after(150, self._poll)

    def _on_event(self, payload):
        job = payload.get("job") or payload.get("job_dir")
        if job:
            path = Path(job)
            self.job_dir = path if path.is_absolute() else self.app_dir / path
        stage = str(payload.get("stage", ""))
        if stage in ("complete", "partial", "cancelled", "error"):
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
        self._refresh_dependent_controls()

    def _pause(self):
        if self.process is None:
            return
        action = "resume" if self.paused else "pause"
        try:
            write_json(self.app_dir / "cache" / "control.json", {"action": action})
        except OSError as exc:
            messagebox.showerror("Could not change processing state", str(exc), parent=self)
            return
        self.paused = not self.paused
        self.pause_button.configure(text="Resume" if self.paused else "Pause")
        self.detail_text.set("Pause requested — the current supported stage or segment will finish first." if self.paused
                             else "Resume requested — processing will continue from the current checkpoint.")
        self._log(self.detail_text.get())

    def _cancel(self):
        if self.process is None:
            return
        try:
            write_json(self.app_dir / "cache" / "control.json", {"action": "cancel"})
        except OSError as exc:
            self.closing = False
            messagebox.showerror("Could not request cancellation", str(exc), parent=self)
            return
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
        self._restore_settings(job)
        try:
            self.last_args = ["make", "--job", str(job), *self._settings_args()]
        except ValueError as exc:
            messagebox.showinfo("Check saved settings", str(exc), parent=self)
            return
        self._load_candidates(force=True)
        self._load_flight_map(force=True)
        self._launch(self.last_args)

    def _restore_settings(self, job):
        settings = read_json(job / "settings.json", {})
        if not isinstance(settings, dict):
            return
        self._restoring_settings = True
        try:
            for key, variable, options, default in (
                    ("duration", self.duration_value, DURATIONS, "auto"),
                    ("style", self.style_value, STYLES, "hype"),
                    ("look", self.look_value, LOOKS, "natural"),
                    ("quality", self.quality_value, QUALITIES, "auto"),
                    ("recognition", self.recognition_value, RECOGNITION_MODES, "auto"),
                    ("edit_order", self.order_value, EDIT_ORDERS, "story"),
                    ("music_end", self.music_end_value, MUSIC_ENDS, "fade"),
                    ("framing", self.framing_value, FRAMINGS, "blur")):
                match = next((label for label, value in options.items() if value == str(settings.get(key, default))), None)
                variable.set(match or next(iter(options)))
            for key, variable, default, low, high in (
                    ("strength", self.strength_value, 0, 0, 1), ("audio_level", self.audio_value, .4, 0, 1),
                    ("music_level", self.music_level_value, .75, 0, 1), ("music_offset", self.music_offset_value, 0, 0, 86400),
                    ("music_fade", self.music_fade_value, 1.5, 0, 30), ("focus_x", self.focus_value, .5, 0, 1),
                    ("recovery", self.recovery_value, 2.5, .5, 8)):
                try:
                    number = float(settings.get(key, default))
                    variable.set(min(high, max(low, number)) if math.isfinite(number) else default)
                except (ValueError, TypeError):
                    variable.set(default)
            self.codec_value.set("H.264" if settings.get("codec") == "h264" else "HEVC")
            self.music_path = Path(settings["music"]) if settings.get("music") else None
            self.beat_value.set(settings.get("beat_sync", True) is not False)
            social = settings.get("social_formats", [])
            social = social.split(",") if isinstance(social, str) else social
            social = social if isinstance(social, (list, tuple)) else []
            for code, variable in self.social_values.items():
                variable.set(code in social)
            paths = settings.get("inputs")
            if isinstance(paths, list):
                self.files, self.folder = [Path(path) for path in paths], None
                self._refresh_inputs()
            if self.quality_value.get() not in self.quality_combo["values"]:
                self.quality_value.set("Auto")
                self._log_warning("The saved enhancement is currently unavailable; Auto is selected.")
            self.finish_value.set(FINISH_MODES[0])
            self.overrides_dirty = False
            self.settings_dirty = False
            self.recognition_dirty = False
        finally:
            self._restoring_settings = False
        self._refresh_dependent_controls()

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
        self._load_flight_map(force=True)
        self._load_job_poster()
        self.progress["value"] = 100
        self.stage_text.set("Your latest edit is ready.")
        self.detail_text.set(f"{self.job_dir.name} · Preview and final playback open your video player.")
        self._refresh_outputs()

    def _render_final(self):
        if not self.job_dir or self.process or self.overrides_dirty or self.settings_dirty:
            return
        try:
            args = ["make", "--job", str(self.job_dir), *self._settings_args()]
        except ValueError as exc:
            messagebox.showinfo("Check your settings", str(exc), parent=self)
            return
        self.last_args = args.copy()
        self._launch(args)

    def _refresh_understanding(self):
        if not self.job_dir or self.process is not None:
            return
        mode = RECOGNITION_MODES.get(self.recognition_value.get())
        if mode is None:
            messagebox.showinfo("Choose recognition", "Choose Automatic, Off, or Thorough before refreshing understanding.", parent=self)
            return
        self._launch(["map-flight", "--job", str(self.job_dir), "--recognition", mode])

    def _regenerate(self):
        if not self.job_dir:
            return
        try:
            args = self._without_options(self._command_args(), {"--job", "--regenerate", "--overrides", "--input", "--folder"})
        except ValueError as exc:
            messagebox.showinfo("Check your settings", str(exc), parent=self)
            return
        args += ["--job", str(self.job_dir), "--regenerate", "--overrides", str(self._save_overrides())]
        self.last_args = args.copy()
        self.overrides_dirty = False
        self.settings_dirty = False
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
        selected_ids = {str(self.candidates[int(index)]["id"]) for index in self.table.selection()
                        if index.isdigit() and int(index) < len(self.candidates)}
        self.candidates = [item for item in candidates if isinstance(item, dict) and item.get("id") is not None]
        timeline = read_json(self.job_dir / "timeline.json", {})
        ordered = {str(shot.get("id")): index for index, shot in enumerate(timeline.get("shots", []))} if isinstance(timeline, dict) else {}
        self.candidates.sort(key=lambda item: (not item.get("selected", False), ordered.get(str(item["id"]), len(ordered)), str(item.get("source", "")), float(item.get("start", 0))))
        self._candidate_mtime = mtime
        ui_path = self.job_dir / "ui-overrides.json"
        rendered_overrides = data.get("overrides") if isinstance(data, dict) else None
        if not isinstance(rendered_overrides, dict):
            rendered_overrides = read_json(self.job_dir / "overrides.json", {})
        rendered_overrides = rendered_overrides if isinstance(rendered_overrides, dict) else {}
        try:
            pending_ui = self.overrides_dirty or ui_path.stat().st_mtime_ns > mtime
        except OSError:
            pending_ui = self.overrides_dirty
        overrides = read_json(ui_path, rendered_overrides) if pending_ui else rendered_overrides
        overrides = overrides if isinstance(overrides, dict) else rendered_overrides
        self.override_choices = {key: [str(value) for value in overrides.get(key, [])] for key in ("keep", "exclude")}
        self.overrides_dirty = any(set(self.override_choices[key]) != set(str(value) for value in rendered_overrides.get(key, [])) for key in ("keep", "exclude"))
        self._paint_candidates()
        self.table.selection_set([str(index) for index, item in enumerate(self.candidates) if str(item["id"]) in selected_ids])
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
            if item.get("trick_label") or item.get("flight_label"):
                label, evidence_state, evidence = self._event_summary(item)
                reason = f"{label} · {evidence_state}" + (f" — {evidence}" if evidence else "")
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
        label, state, evidence = self._event_summary(item)
        body = (f"{Path(item.get('source', '')).name} · {timestamp(item.get('start'))}–{timestamp(item.get('end'))}\n\n"
                f"{label} · {state}\n\n{evidence}")
        messagebox.showinfo("Source moment", body, parent=self)

    def _open_selected_source(self):
        selected = self.table.selection()
        if selected:
            item = self.candidates[int(selected[0])]
            path = Path(item.get("source", item.get("source_path", "")))
            if path.is_file():
                self._open_path(path)
                self.review_text.set(f"Source opened in your video player. Selected range: {timestamp(item.get('start'))}–{timestamp(item.get('end'))}.")

    @staticmethod
    def _parse_time(value):
        try:
            parts = [float(part) for part in str(value).strip().split(":")]
            if not 1 <= len(parts) <= 3 or any(not math.isfinite(part) or part < 0 for part in parts):
                raise ValueError
            if len(parts) > 1 and any(part >= 60 for part in parts[1:]):
                raise ValueError
            return sum(part * 60 ** index for index, part in enumerate(reversed(parts)))
        except ValueError:
            raise ValueError("Enter time as seconds, minutes:seconds, or hours:minutes:seconds.") from None

    def _source_identity(self, source, candidate=None):
        probes = read_json(self.job_dir / "sources.json", []) if self.job_dir else []
        probe = next((item for item in probes if isinstance(item, dict) and item.get("source") == source), {})
        identity = probe.get("sha256") or probe.get("identity")
        if isinstance(identity, dict):
            identity = identity.get("sha256")
        if candidate is not None:
            candidate_identity = candidate.get("identity") if candidate.get("source") == source else None
            if isinstance(candidate_identity, dict):
                candidate_identity = candidate_identity.get("sha256")
            if not isinstance(candidate_identity, str) or len(candidate_identity) != 64:
                raise ValueError("This displayed moment has no verified recording identity. Regenerate and reload its analysis before teaching it.")
            if identity and candidate_identity.lower() != str(identity).lower():
                raise ValueError("This displayed moment belongs to an earlier version of the recording. Regenerate and reload its analysis before teaching it.")
            identity = identity or candidate_identity
        if not isinstance(identity, str) or len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity.lower()):
            raise ValueError("This recording has no verified file identity. Regenerate its analysis before teaching or saving an exact range.")
        try:
            stat = Path(source).stat()
        except OSError:
            raise ValueError("This recording is missing. Restore it or analyze the current recording first.") from None
        if (probe.get("size_bytes") is not None and stat.st_size != probe["size_bytes"]) or (
                probe.get("mtime_ns") is not None and stat.st_mtime_ns != probe["mtime_ns"]):
            raise ValueError("This recording changed after analysis. Regenerate its analysis before teaching or saving an exact range.")
        return identity.lower()

    def _save_exact_range(self, source, start, end, reason="User-entered exact source range"):
        if not self.job_dir:
            raise ValueError("Analyze a session or open a saved job first.")
        sources = read_json(self.job_dir / "sources.json", [])
        match = next((item for item in sources if isinstance(item, dict) and item.get("source") == source), None)
        if not match or not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end <= float(match["duration"]):
            raise ValueError("The range must stay within a recording in this job, with the end after the start.")
        identity = self._source_identity(source)
        path = self.job_dir / "reviewed-intervals.json"
        reviews = read_json(path, None if path.exists() else [])
        if not isinstance(reviews, list):
            raise ValueError("The saved reviewed-ranges file needs repair before adding a range.")
        record = {"source": source, "source_identity": identity, "start": start, "end": end, "key": "user_range_" + uuid.uuid4().hex[:12],
                  "reason": reason, "confidence": 1.0, "review_method": "User-entered exact source range", "keep": True}
        existing = next((item for item in reviews if isinstance(item, dict) and all(item.get(key) == record[key] for key in ("source", "start", "end"))), None)
        if existing is not None:
            existing.update(record)
        else:
            reviews.append(record)
        write_json(path, reviews)
        self.settings_dirty = True
        self.review_text.set("Exact range saved and marked to keep. Regenerate the edit to include it.")
        self._refresh_outputs()
        return record

    def _add_exact_range(self):
        if not self.job_dir:
            return
        sources = read_json(self.job_dir / "sources.json", [])
        sources = [item for item in sources if isinstance(item, dict) and item.get("source") and item.get("duration")]
        if not sources:
            self.review_text.set("Analyze the recordings first so exact ranges can be checked against their lengths.")
            return
        dialog = tk.Toplevel(self)
        dialog.title("Keep an exact source range")
        dialog.transient(self)
        dialog.resizable(False, False)
        pane = ttk.Frame(dialog, padding=18)
        pane.pack(fill="both", expand=True)
        paths = [item["source"] for item in sources]
        selected = self.table.selection()
        item = self.candidates[int(selected[0])] if selected else {}
        source_value = tk.StringVar(value=item.get("source") if item.get("source") in paths else paths[0])
        start_value = tk.StringVar(value=str(item.get("start", 0)))
        end_value = tk.StringVar(value=str(item.get("end", min(10, sources[0]["duration"])) ))
        ttk.Label(pane, text="Recording").grid(row=0, column=0, sticky="w", pady=6)
        ttk.Combobox(pane, textvariable=source_value, values=paths, state="readonly", width=79).grid(row=0, column=1, sticky="ew")
        for row, label, variable in ((1, "Start", start_value), (2, "End", end_value)):
            ttk.Label(pane, text=label).grid(row=row, column=0, sticky="w", pady=6, padx=(0, 12))
            ttk.Entry(pane, textvariable=variable, width=20).grid(row=row, column=1, sticky="w")
        ttk.Label(pane, text="Use seconds or mm:ss. Keep the complete approach, trick and continued flight.",
                  style="Muted.TLabel").grid(row=3, column=0, columnspan=2, sticky="w", pady=8)
        def save():
            try:
                self._save_exact_range(source_value.get(), self._parse_time(start_value.get()), self._parse_time(end_value.get()))
            except (ValueError, OSError) as exc:
                messagebox.showinfo("Check this range", str(exc), parent=dialog)
                return
            dialog.destroy()
        ttk.Button(pane, text="Save range to keep", style="Primary.TButton", command=save).grid(row=4, column=1, sticky="e", pady=(7, 0))
        dialog.grab_set()

    def _teach_moment(self):
        selected = self.table.selection()
        if not selected or not self.job_dir:
            self.review_text.set("Select one or more moments, choose their label, then Teach this moment.")
            return
        label = self.flight_label_value.get()
        if label not in FLIGHT_LABELS:
            return
        path = self.job_dir / "flight-labels.json"
        labels = read_json(path, None if path.exists() else [])
        review_path = self.job_dir / "reviewed-intervals.json"
        reviews = read_json(review_path, None if review_path.exists() else [])
        if not isinstance(labels, list) or not isinstance(reviews, list):
            messagebox.showinfo("Check saved examples", "The saved labels or ranges file needs repair before adding an example.", parent=self)
            return
        try:
            identities = {index: self._source_identity(self.candidates[int(index)]["source"], self.candidates[int(index)]) for index in selected}
        except ValueError as exc:
            messagebox.showinfo("Analyze this recording first", str(exc), parent=self)
            return
        for index in selected:
            item = self.candidates[int(index)]
            bounds = {"source": item["source"], "start": item["start"], "end": item["end"]}
            def matches(record):
                return isinstance(record, dict) and all(record.get(key) == value for key, value in bounds.items())
            labels = [record for record in labels if not matches(record)]
            labels.append({**bounds, "source_identity": identities[index], "label": label, "confidence": 1.0})
            existing = next((record for record in reviews if matches(record)), None)
            if existing is not None:
                existing["label"] = label
                existing["source_identity"] = identities[index]
            else:
                reviews.append({**bounds, "source_identity": identities[index], "key": "user_label_" + uuid.uuid4().hex[:12], "reason": f"User-labeled {label}",
                                "label": label, "confidence": 1.0, "keep": False, "review_method": "User-labeled source moment"})
        write_json(path, labels)
        write_json(review_path, reviews)
        self.settings_dirty = True
        self.review_text.set(f"Saved {len(selected)} optional example(s) as {label}. Use Refresh understanding on the Flight map to update recognition.")
        self._refresh_outputs()

    def _load_flight_map(self, force=False):
        if not self.job_dir:
            return
        path = self.job_dir / "flight-map.json"
        try:
            key = (str(path), path.stat().st_mtime_ns)
        except OSError:
            if force:
                self.flight_table.delete(*self.flight_table.get_children())
                self._flight_data = {}
                self._flight_rows = []
                self._paint_flight_timeline()
                self._refresh_watch_section()
                self.learning_text.set("This job has no flight map yet. Use Refresh understanding to add video estimates. Local examples are optional.")
            return
        if not force and key == self._flight_mtime:
            return
        data = read_json(path, {})
        if not isinstance(data, dict):
            return
        self._flight_mtime = key
        self._flight_data = data
        learning = data.get("learning", {})
        learning = learning if isinstance(learning, dict) else {}
        video = learning.get("video_model", {})
        video = video if isinstance(video, dict) else {}
        online = learning.get("online_model", {})
        if video.get("mode") == "off":
            summary = "Video recognition is off. Motion estimates remain available."
        elif video.get("available"):
            mode = next((name for name, value in RECOGNITION_MODES.items() if value == video.get("mode")), "Automatic")
            coverage = video.get("coverage_seconds")
            coverage_text = f" · {coverage:.0f}s covered" if isinstance(coverage, (int, float)) and math.isfinite(coverage) else ""
            summary = (f"Online-trained video · {video.get('name', 'installed model')} · {mode}. "
                       f"{video.get('windows_analyzed', 0)} windows reviewed{coverage_text}. Trick labels remain estimates.")
        elif video:
            summary = "Video recognition is unavailable for this job. " + self._flight_status_message(video.get("message", "Install the video model, then refresh understanding."))
        elif isinstance(online, dict) and online.get("available"):
            summary = "Scene context is available. Use Refresh understanding to add video estimates with the installed model."
        else:
            summary = self._flight_status_message(learning.get("message", "Video recognition is not available for this job yet."))
        if video.get("available") and video.get("message"):
            summary += " " + self._flight_status_message(video["message"])
        self.learning_text.set(summary + f" {learning.get('examples', 0)} optional local examples.")
        self._paint_flight_rows()

    def _flight_status_message(self, value):
        text = str(value).strip()
        if len(text) <= 160 and "\n" not in text and "\r" not in text:
            return text
        # Diagnostics must remain accessible without expanding this fixed header
        # until the timelines and results disappear below the window.
        self._log_warning("Flight understanding details: " + text)
        first_line = text.splitlines()[0].strip()
        brief = re.split(r"(?<=[.!?])\s+", first_line, maxsplit=1)[0]
        if len(brief) > 160:
            brief = brief[:157].rstrip() + "…"
        return brief + " See Activity for details."

    @staticmethod
    def _plain_text(value):
        if value is None:
            return ""
        if isinstance(value, list):
            return "; ".join(SeshApp._plain_text(item) for item in value)
        if isinstance(value, dict):
            return "; ".join(f"{key}: {SeshApp._plain_text(item)}" for key, item in value.items())
        return str(value)

    @staticmethod
    def _event_summary(event):
        method = (event.get("trick_method") or event.get("method") or
                  (event.get("flight_method", "") if not event.get("trick_label") else ""))
        status = event.get("trick_status") or event.get("status", "")
        confirmed = event.get("flight_method") == "user-confirmed" or method == "user-confirmed" or status == "confirmed"
        label = ((event.get("flight_label") if confirmed else None) or event.get("trick_label") or
                 event.get("label") or event.get("flight_label") or "Motion estimate")
        state = ("Confirmed" if confirmed else "Uncertain" if status == "uncertain" or
                 str(label).lower() in ("uncertain", "unknown", "uncertain motion", "unmeasured interval") else "Estimate")
        descriptions = []
        model = event.get("trick_model") or event.get("model")
        if confirmed:
            descriptions.append("User confirmation")
        elif method == "online-pretrained video model":
            descriptions.append("Online-trained video" + (f" · {model}" if model else ""))
        elif method:
            descriptions.append(str(method))
            if model:
                descriptions.append("Video model context · " + str(model))
        elif model:
            descriptions.append("Video model context · " + str(model))
        elif event.get("trick_label"):
            descriptions.append("Recognition estimate")
        for key in ("trick_evidence", "evidence", "reason", "flight_reason"):
            text = SeshApp._plain_text(event.get(key))
            if text and text not in descriptions:
                descriptions.append(text)
        scene = event.get("scene") or event.get("scene_context")
        if isinstance(scene, dict) and scene.get("label") in ("woodland", "park or open grass", "cultivated field", "sky", "built surroundings", "water"):
            descriptions.append("Scene estimate: " + scene["label"])
        raw_label = event.get("trick_raw_label") or event.get("raw_label")
        if not confirmed and raw_label and raw_label != label:
            descriptions.append("Video model originally reported: " + str(raw_label))
        for key in ("trick_checks", "checks"):
            checks = SeshApp._plain_text(event.get(key))
            if checks and "Checks: " + checks not in descriptions:
                descriptions.append("Checks: " + checks)
        return str(label), state, "; ".join(descriptions)

    def _paint_flight_rows(self):
        if not hasattr(self, "flight_table"):
            return
        selected = self._selected_flight_event()
        selected_key = self._flight_event_key(*selected) if selected else None
        self.flight_table.delete(*self.flight_table.get_children())
        self._flight_rows = []
        chosen_filter = self.flight_filter_value.get()
        for source in self._flight_data.get("sources", []):
            combined = list(source.get("events", [])) + [{**event, "_video_origin": True} for event in source.get("video_events", [])]
            seen = set()
            for event in sorted(combined, key=lambda row: (row.get("start", 0), row.get("end", 0))):
                label, state, evidence = self._event_summary(event)
                status = event.get("trick_status") or event.get("status")
                name = label.lower().replace("_", " ")
                if chosen_filter == "Possible tricks" and not (status == "suggested" and name in ("roll", "flip", "split-s", "powerloop")):
                    continue
                if chosen_filter == "Ordinary flight" and not (state != "Uncertain" and name in
                        ("ordinary flight", "tree weaving", "dive", "orbit", "moving flight line estimate", "close-pass / weave estimate", "smooth line")):
                    continue
                if chosen_filter == "Uncertain" and state != "Uncertain":
                    continue
                identity = (event.get("start"), event.get("end"), label, state, event.get("model"), event.get("method"))
                if identity in seen:
                    continue
                seen.add(identity)
                index = len(self._flight_rows)
                self._flight_rows.append((source, event))
                self.flight_table.insert("", "end", iid=str(index), values=(Path(source.get("source", "")).name,
                    timestamp(event.get("start")), timestamp(event.get("end")), label, state, evidence))
                if selected_key is not None and self._flight_event_key(source, event) == selected_key:
                    self.flight_table.selection_set(str(index))
        self._paint_flight_timeline()
        self._refresh_watch_section()

    def _paint_flight_timeline(self):
        if not hasattr(self, "flight_canvas"):
            return
        canvas = self.flight_canvas
        canvas.delete("all")
        sources = self._flight_data.get("sources", [])
        if not sources:
            canvas.configure(height=96)
            canvas.create_text(18, 42, text="Your recordings will appear as time-based flight lines.",
                               fill=MUTED, font=("Segoe UI", 10), anchor="w")
            return
        canvas.configure(height=50 + 36 * len(sources))
        palette = {"Flight": "#78a99a", "Rotation": LIME, "Close pass": "#d7ae74", "Idle / arrival": "#64716b",
                   "Uncertain": "#aab0b6"}
        for index, (label, color) in enumerate(palette.items()):
            x = 16 + index * 142
            canvas.create_rectangle(x, 13, x + 7, 20, fill=color, outline="")
            canvas.create_text(x + 14, 17, text=label, fill=MUTED, anchor="w", font=("Segoe UI", 8))
        canvas.create_text(16, 32, text="Upper band: motion estimates · Lower band: video observations",
                           fill=MUTED, anchor="w", font=("Segoe UI", 8))
        longest = max(float(item.get("duration", 1)) for item in sources) or 1
        left, width = 132, max(120, canvas.winfo_width() - 195)
        for index, source in enumerate(sources):
            y = 42 + index * 36
            duration = float(source.get("duration", 0))
            end_x = left + width * duration / longest
            canvas.create_text(14, y + 9, text=Path(source.get("source", "")).name, fill=INK, anchor="w", font=("Segoe UI", 8))
            for offset in (0, 11):
                canvas.create_rectangle(left, y+offset, end_x, y+offset+8, fill=FIELD, outline="")
            rows = [(row, event) for row, (recording, event) in enumerate(self._flight_rows) if recording is source]
            for row_id, event in sorted(rows, key=lambda item: bool(item[1].get("_video_origin"))):
                label, state, _ = self._event_summary(event)
                label = label.lower()
                color = palette["Uncertain"] if state == "Uncertain" else \
                        palette["Idle / arrival"] if any(word in label for word in ("idle", "arrival", "ground", "landing", "crash")) else \
                        palette["Rotation"] if any(word in label for word in ("rotat", "flip", "roll", "loop", "split")) else \
                        palette["Close pass"] if any(word in label for word in ("weav", "pass", "proxim")) else palette["Flight"]
                x0 = left + width * max(0, float(event.get("start", 0))) / longest
                x1 = left + width * min(duration, float(event.get("end", 0))) / longest
                band = "video-event" if event.get("_video_origin") else "motion-event"
                band_y = y + (11 if event.get("_video_origin") else 0)
                rect = canvas.create_rectangle(x0, band_y, max(x0+1, x1), band_y+8,
                                               fill=color, outline=SURFACE, width=1, tags=(band, f"flight-row-{row_id}"))
                canvas.tag_bind(rect, "<Button-1>", lambda _, row=row_id: self._select_flight_event(row))
            canvas.create_text(end_x+8, y+9, text=timestamp(duration), fill=MUTED, anchor="w", font=("Segoe UI", 8))

    def _select_flight_event(self, index):
        rows = self.flight_table.get_children()
        if 0 <= index < len(rows):
            self.flight_table.selection_set(rows[index])
            self.flight_table.see(rows[index])
            self._refresh_watch_section()

    @staticmethod
    def _flight_event_key(source, event):
        return (source.get("source"), source.get("identity"), event.get("start"), event.get("end"),
                event.get("label"), event.get("trick_label"), event.get("method"), event.get("trick_method"))

    def _selected_flight_event(self):
        selected = self.flight_table.selection()
        if not selected:
            return None
        try:
            index = int(selected[0])
        except (TypeError, ValueError):
            return None
        return self._flight_rows[index] if 0 <= index < len(self._flight_rows) else None

    def _refresh_watch_section(self):
        if not hasattr(self, "watch_section_button"):
            return
        available = bool(self.job_dir and self._selected_flight_event() is not None
                         and (self.process is None or self.mapping_only))
        self.watch_section_button.configure(state="normal" if available else "disabled")

    def _watch_flight_section(self):
        selected = self._selected_flight_event()
        if not self.job_dir or not selected or (self.process is not None and not self.mapping_only):
            return
        source, event = selected
        try:
            from .source_review import play_section
            result = play_section(source.get("source", ""), event.get("start"), event.get("end"),
                                  source_duration=source.get("duration"), context=2.0, app_dir=self.app_dir)
        except (ValueError, OSError, RuntimeError) as exc:
            messagebox.showerror("Could not watch this section", str(exc), parent=self)
            return
        self.detail_text.set(f"Watching {Path(result['source']).name} · {timestamp(result['start'])}–{timestamp(result['end'])} with entry and exit context.")

    def _show_flight_event(self, _event=None):
        selected = self.flight_table.selection()
        if not selected:
            return
        index = int(selected[0])
        if 0 <= index < len(self._flight_rows):
            source, event = self._flight_rows[index]
            label, state, evidence = self._event_summary(event)
            body = f"{Path(source.get('source', '')).name} · {timestamp(event.get('start'))}–{timestamp(event.get('end'))}\n\n" \
                   f"{label} · {state}\n\n{evidence}"
            messagebox.showinfo("Flight event", body, parent=self)

    def _set_choice(self, choice):
        selected = self.table.selection()
        if not selected:
            self.review_text.set("Select one or more moment rows first. Use Ctrl or Shift to select several.")
            return
        try:
            self._update_review_keep([(self.candidates[int(index)], choice == "keep") for index in selected])
        except (ValueError, OSError) as exc:
            messagebox.showinfo("Check saved ranges", str(exc), parent=self)
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
        try:
            self._update_review_keep(clear=True)
        except (ValueError, OSError) as exc:
            messagebox.showinfo("Check saved ranges", str(exc), parent=self)
            return
        self.override_choices = {"keep": [], "exclude": []}
        self.overrides_dirty = True
        self._save_overrides()
        self._paint_candidates()
        self.review_text.set("Overrides cleared. Regenerate edit to return to automatic selection.")

    def _update_review_keep(self, choices=(), clear=False):
        if not self.job_dir:
            return
        path = self.job_dir / "reviewed-intervals.json"
        if not path.is_file():
            return
        reviews = read_json(path)
        if not isinstance(reviews, list):
            raise ValueError("The saved reviewed-ranges file needs repair before changing its keep settings.")
        changed = False
        for review in reviews:
            if not isinstance(review, dict):
                continue
            if clear and review.get("keep"):
                review["keep"] = False
                changed = True
            for candidate, keep in choices:
                matches = (candidate.get("review_key") and candidate["review_key"] == review.get("key")) or all(
                    candidate.get(key) == review.get(key) for key in ("source", "start", "end"))
                if matches and review.get("keep") is not keep:
                    review["keep"] = keep
                    changed = True
        if changed:
            write_json(path, reviews)

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
        self.make_button.configure(state="normal" if not busy and (self.files or self.folder) else "disabled")
        preview = bool(self.job_dir and (self.job_dir / "preview.mp4").is_file())
        current_pending = busy or self.overrides_dirty or self.settings_dirty
        status = read_json(self.job_dir / "status.json", {}) if self.job_dir else {}
        completed_final = bool(self.job_dir and (self.job_dir / "final_4k.mp4").is_file()
                               and isinstance(status, dict) and status.get("stage") == "complete")
        self.preview_button.configure(state="normal" if preview else "disabled")
        self.play_final_button.configure(state="normal" if completed_final and not current_pending else "disabled")
        self.final_button.configure(state="normal" if preview and not current_pending else "disabled")
        editable = bool(self.job_dir and self.candidates and not busy)
        for button in (self.keep_button, self.exclude_button, self.reset_button, self.source_button, self.teach_button):
            button.configure(state="normal" if editable else "disabled")
        job_editable = bool(self.job_dir and not busy)
        self.map_button.configure(state="normal" if job_editable else "disabled")
        self._refresh_watch_section()
        self.regenerate_button.configure(state="normal" if job_editable else "disabled")
        self.range_button.configure(state="normal" if job_editable and (self.job_dir / "sources.json").is_file() else "disabled")
        paths = {}
        if self.job_dir:
            for code, (label, _) in SOCIAL_FORMATS.items():
                final = self.job_dir / "social" / f"{code}.mp4"
                social_preview = self.job_dir / "social-preview" / f"{code}.mp4"
                if final.is_file() and isinstance(status, dict) and status.get("stage") == "complete":
                    paths[f"{label} — final"] = final
                if social_preview.is_file():
                    paths[f"{label} — preview"] = social_preview
        if paths != self._playback_paths:
            self._playback_paths = paths
            self.social_play_combo.configure(values=list(paths))
            if self.social_play_value.get() not in paths:
                self.social_play_value.set(next(iter(paths), ""))
        self.social_play_button.configure(state="normal" if paths and not current_pending else "disabled")
        self.social_play_combo.configure(state="readonly" if paths and not current_pending else "disabled")
        if current_pending:
            self.social_ready_text.set("Render or regenerate the current settings before playing social exports.")
        elif paths:
            self.social_ready_text.set("Only rendered files are listed. Playback opens your Windows video player.")
        else:
            self.social_ready_text.set("Choose export sizes above; their previews and finals appear after rendering.")

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
        self.gpu_text.set(f"Saved hardware check — GPU: {gpu}  •  Encoder: {encoder}. Checked again when processing starts.")
        detected = str(data.get("gpu_name", "")).strip().lower() not in ("", "not detected", "none", "unknown")
        self.gpu_status.set("GPU detected" if detected else "CPU / fallback")
        self._refresh_optional_controls()
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

    def _play_social(self):
        self._refresh_outputs()
        path = self._playback_paths.get(self.social_play_value.get())
        if path and self.social_play_button.instate(["!disabled"]):
            self._open_path(path)

    def _open_output(self):
        path = self.job_dir or self.app_dir / "output"
        path.mkdir(parents=True, exist_ok=True)
        self._open_path(path)

    def _close(self):
        if self.process:
            if messagebox.askyesno("Analysis in progress" if self.mapping_only else "Render in progress", "Cancel at the next safe boundary and close? Completed work will be kept.", parent=self):
                self.closing = True
                self._cancel()
        else:
            self.destroy()

    def destroy(self):
        self.closing = True
        callback = getattr(self, "_poll_after", None)
        if callback:
            try:
                self.after_cancel(callback)
            except tk.TclError:
                pass
            self._poll_after = None
        super().destroy()

    def report_callback_exception(self, exc, value, tb):
        details = "".join(traceback.format_exception(exc, value, tb))
        self._log(details)
        log_path = self.app_dir / "logs" / "ui-error.log"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(details + "\n")
            location = "Details were saved in logs/ui-error.log."
        except OSError:
            location = "The diagnostic file could not be saved. Details remain in Activity; check folder write permissions."
        messagebox.showerror("FPV Sesh", f"{value}\n\n{location}", parent=self)


def main():
    app = SeshApp()
    app.mainloop()


if __name__ == "__main__":
    main()
