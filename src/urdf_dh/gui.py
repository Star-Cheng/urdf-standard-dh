"""Native desktop workflow for local URDF -> standard DH / Craig MDH."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import queue
import re
import sys
import threading
import tkinter as tk
from tkinter import filedialog, font, messagebox, ttk
from typing import Any, Sequence

from . import __version__
from .gui_controller import Document, convert_document, load_document, parse_options, save_result


BACKGROUND = "#f2f5f7"
SURFACE = "#ffffff"
INK = "#182b35"
MUTED = "#637682"
ACCENT = "#087f83"
BORDER = "#dce5e9"
CONVENTION_LABELS = {"标准 DH": "standard", "MDH（Craig）": "modified"}


def demo_path() -> Path:
    """The wheel includes the same read-only example as the source project."""
    return Path(__file__).with_name("resources") / "mixed_arm.urdf"


class DHApp:
    """Tk stays on the main thread; workers exchange plain results via a queue."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.document: Document | None = None
        self.result = None
        self.busy = False
        self._closed = False
        self._generation = 0
        self._suspend_changes = True
        self._queue: queue.Queue = queue.Queue()
        self._after_id: str | None = None
        self._last_export_dir: Path | None = None

        root.title(f"URDF DH / MDH 提取工具 · {__version__}")
        root.configure(background=BACKGROUND)
        width, height = 1280, 840
        root.geometry(f"{width}x{height}+{max(0, (root.winfo_screenwidth()-width)//2)}"
                      f"+{max(0, (root.winfo_screenheight()-height)//2)}")
        root.minsize(1060, 740)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._configure_style()

        self.base_var = tk.StringVar(root)
        self.tip_var = tk.StringVar(root)
        self.convention_var = tk.StringVar(root, "标准 DH")
        self.formula_var = tk.StringVar(root)
        self.samples_var = tk.StringVar(root, "100")
        self.seed_var = tk.StringVar(root, "0")
        self.position_tol_var = tk.StringVar(root, "1e-7")
        self.rotation_tol_var = tk.StringVar(root, "1e-7")
        self.parallel_tol_var = tk.StringVar(root, "1e-12")
        self.path_var = tk.StringVar(root, "尚未选择文件")
        self.file_info_var = tk.StringVar(root, "支持已展开的 .urdf / .xml 文件")
        self.status_var = tk.StringVar(root, "选择一个 URDF 文件开始。")
        self.chain_var = tk.StringVar(root, "指定起点和终点，构造完整运动链")
        self.badge_var = tk.StringVar(root, "等待导入")
        self.advanced_var = tk.BooleanVar(root, False)
        self.stat_vars = [tk.StringVar(root, "—") for _ in range(4)]
        self._build()
        self._update_convention_display()
        for variable in (self.base_var, self.tip_var, self.convention_var, self.samples_var, self.seed_var,
                         self.position_tol_var, self.rotation_tol_var, self.parallel_tol_var):
            variable.trace_add("write", self._inputs_changed)
        self._suspend_changes = False
        self._refresh_actions()
        self._after_id = root.after(60, self._poll_worker)
        root.bind("<Control-o>", lambda _event: self.choose_file())
        root.bind("<Control-Return>", lambda _event: self.run_conversion())

    def _configure_style(self) -> None:
        families = set(font.families(self.root))
        family = next((name for name in ("Noto Sans CJK SC", "WenQuanYi Micro Hei",
                       "Microsoft YaHei", "PingFang SC", "Droid Sans Fallback")
                       if name in families), "DejaVu Sans")
        self.ui_font = family
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            font.nametofont(name).configure(family=family, size=10)
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background=BACKGROUND)
        style.configure("Card.TFrame", background=SURFACE)
        style.configure("TLabel", background=SURFACE, foreground=INK, font=(family, 10))
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("Section.TLabel", font=(family, 11, "bold"))
        style.configure("Title.TLabel", font=(family, 20, "bold"))
        style.configure("Value.TLabel", font=(family, 17, "bold"))
        style.configure("TButton", font=(family, 10), padding=(12, 8), background="#eaf0f3",
                        foreground=INK, borderwidth=0)
        style.map("TButton", background=[("active", "#dce8ec"), ("disabled", "#edf1f3")],
                  foreground=[("disabled", "#9aa7ae")])
        style.configure("Accent.TButton", background=ACCENT, foreground="white",
                        padding=(14, 11), font=(family, 11, "bold"))
        style.map("Accent.TButton", background=[("active", "#06696c"), ("disabled", "#cbdcde")],
                  foreground=[("disabled", "#718e90")])
        style.configure("TCombobox", padding=7, arrowsize=15, fieldbackground="white")
        style.configure("TEntry", padding=7, fieldbackground="white")
        style.configure("Treeview", font=(family, 10), rowheight=34, background="white",
                        fieldbackground="white", foreground=INK, borderwidth=0)
        style.configure("Treeview.Heading", font=(family, 9, "bold"), background="#eaf0f3",
                        foreground=MUTED, padding=(7, 9), relief="flat")
        style.map("Treeview", background=[("selected", "#d6efed")],
                  foreground=[("selected", INK)])
        style.configure("TNotebook", background=SURFACE, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(13, 9), background="#edf2f4", foreground=MUTED)
        style.map("TNotebook.Tab", background=[("selected", SURFACE)],
                  foreground=[("selected", ACCENT)])
        style.configure("TCheckbutton", background=SURFACE, foreground=MUTED)
        style.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor="#e5efef",
                        borderwidth=0, thickness=3)

    def _build(self) -> None:
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)
        header = ttk.Frame(self.root, style="Card.TFrame", padding=(26, 17))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="ROBOTICS  /  KINEMATICS", foreground=ACCENT,
                  font=(self.ui_font, 9, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="URDF DH / MDH 提取工具", style="Title.TLabel").grid(
            row=1, column=0, sticky="w", pady=(3, 0))
        ttk.Label(header, text="选择模型 · 指定运动链 · 验证并导出", style="Muted.TLabel").grid(
            row=2, column=0, sticky="w", pady=(4, 0))
        tk.Label(header, text="● 本地运行   ·   m / rad", background="#e7f3f1",
                 foreground=ACCENT, font=(self.ui_font, 10), padx=13, pady=8).grid(
            row=0, column=1, rowspan=3, sticky="e")

        main = ttk.Frame(self.root, padding=(22, 18, 22, 10))
        main.grid(row=1, column=0, sticky="nsew")
        main.rowconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        sidebar_shell = ttk.Frame(main, style="Card.TFrame")
        sidebar_shell.grid(row=0, column=0, sticky="nsw", padx=(0, 18))
        sidebar_shell.rowconfigure(0, weight=1)
        sidebar_shell.columnconfigure(0, weight=1)
        self.sidebar_canvas = tk.Canvas(sidebar_shell, background=SURFACE, width=302,
                                        highlightthickness=0, borderwidth=0)
        self.sidebar_canvas.grid(row=0, column=0, sticky="nsew")
        side_scrollbar = ttk.Scrollbar(sidebar_shell, orient="vertical", command=self.sidebar_canvas.yview)
        side_scrollbar.grid(row=0, column=1, sticky="ns")
        self.sidebar_canvas.configure(yscrollcommand=side_scrollbar.set)
        sidebar = ttk.Frame(self.sidebar_canvas, style="Card.TFrame", padding=(20, 18, 14, 6))
        self._sidebar_window = self.sidebar_canvas.create_window((0, 0), window=sidebar, anchor="nw")
        sidebar.bind("<Configure>", lambda _event: self.sidebar_canvas.configure(
            scrollregion=self.sidebar_canvas.bbox("all")))
        self.sidebar_canvas.bind("<Configure>", lambda event: self.sidebar_canvas.itemconfigure(
            self._sidebar_window, width=event.width))
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.root.bind(sequence, self._scroll_sidebar, add="+")
        sidebar.columnconfigure(0, weight=1)
        ttk.Label(sidebar, text="01   导入模型", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        self.file_button = ttk.Button(sidebar, text="选择 URDF 文件…", command=self.choose_file,
                                      style="Accent.TButton")
        self.file_button.grid(row=1, column=0, sticky="ew", pady=(13, 10))
        ttk.Label(sidebar, textvariable=self.path_var, wraplength=270,
                  font=(self.ui_font, 9), foreground=INK).grid(row=2, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self.file_info_var, style="Muted.TLabel", wraplength=270,
                  font=(self.ui_font, 9)).grid(row=3, column=0, sticky="w", pady=(5, 8))
        self.demo_button = ttk.Button(sidebar, text="加载内置示例", command=self.load_demo)
        self.demo_button.grid(row=4, column=0, sticky="w")
        ttk.Separator(sidebar).grid(row=5, column=0, sticky="ew", pady=18)
        chain_frame = ttk.Frame(sidebar, style="Card.TFrame")
        chain_frame.grid(row=6, column=0, sticky="ew")
        chain_frame.columnconfigure(0, weight=1)
        ttk.Label(chain_frame, text="02   选择运动链", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(chain_frame, text="Base link  ·  起点", style="Muted.TLabel").grid(
            row=1, column=0, sticky="w", pady=(12, 5))
        self.base_combo = ttk.Combobox(chain_frame, textvariable=self.base_var, state="disabled", width=26)
        self.base_combo.grid(row=2, column=0, sticky="ew")
        ttk.Label(chain_frame, text="Tip link  ·  终点", style="Muted.TLabel").grid(
            row=3, column=0, sticky="w", pady=(10, 5))
        self.tip_combo = ttk.Combobox(chain_frame, textvariable=self.tip_var, state="disabled", width=26)
        self.tip_combo.grid(row=4, column=0, sticky="ew")
        ttk.Separator(sidebar).grid(row=7, column=0, sticky="ew", pady=18)
        options = ttk.Frame(sidebar, style="Card.TFrame")
        options.grid(row=8, column=0, sticky="ew")
        options.columnconfigure(0, weight=1)
        options.columnconfigure(1, weight=1)
        ttk.Label(options, text="随机验证数", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(options, text="随机种子", style="Muted.TLabel").grid(row=0, column=1, sticky="w", padx=(10, 0))
        ttk.Entry(options, textvariable=self.samples_var, width=12).grid(row=1, column=0, sticky="ew", pady=(5, 8))
        ttk.Entry(options, textvariable=self.seed_var, width=12).grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=(5, 8))
        ttk.Checkbutton(options, text="高级验证参数", variable=self.advanced_var,
                        command=self._toggle_advanced).grid(row=2, column=0, columnspan=2, sticky="w")
        self.advanced_frame = ttk.Frame(options, style="Card.TFrame")
        self.advanced_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.advanced_frame.columnconfigure(1, weight=1)
        for row, (label, variable) in enumerate([
            ("位置容差 (m)", self.position_tol_var), ("旋转容差 (rad)", self.rotation_tol_var),
            ("平行轴阈值", self.parallel_tol_var),
        ]):
            ttk.Label(self.advanced_frame, text=label, style="Muted.TLabel",
                      font=(self.ui_font, 9)).grid(row=row, column=0, sticky="w", pady=3)
            ttk.Entry(self.advanced_frame, textvariable=variable, width=12).grid(
                row=row, column=1, sticky="ew", padx=(10, 0), pady=3)
        self.advanced_frame.grid_remove()
        controls = ttk.Frame(sidebar_shell, style="Card.TFrame", padding=(20, 12, 20, 16))
        controls.grid(row=1, column=0, columnspan=2, sticky="sew")
        controls.columnconfigure(0, weight=1)
        convention_frame = ttk.Frame(controls, style="Card.TFrame")
        convention_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        convention_frame.columnconfigure(1, weight=1)
        ttk.Label(convention_frame, text="提取模式", style="Muted.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 10))
        self.convention_combo = ttk.Combobox(convention_frame, textvariable=self.convention_var,
            values=tuple(CONVENTION_LABELS), state="readonly", width=16)
        self.convention_combo.grid(row=0, column=1, sticky="ew")
        self.run_button = ttk.Button(controls, text="提取并验证 DH", style="Accent.TButton",
                                     command=self.run_conversion)
        self.run_button.grid(row=1, column=0, sticky="ew")
        self.progress = ttk.Progressbar(controls, mode="indeterminate")
        self.progress.grid(row=2, column=0, sticky="ew", pady=(8, 6))
        ttk.Label(controls, textvariable=self.status_var, wraplength=270, style="Muted.TLabel",
                  font=(self.ui_font, 9)).grid(row=3, column=0, sticky="w")

        results = ttk.Frame(main, style="Card.TFrame", padding=20)
        results.grid(row=0, column=1, sticky="nsew")
        results.columnconfigure(0, weight=1)
        results.rowconfigure(3, weight=1)
        title = ttk.Frame(results, style="Card.TFrame")
        title.grid(row=0, column=0, sticky="ew")
        title.columnconfigure(0, weight=1)
        ttk.Label(title, text="提取结果", font=(self.ui_font, 14, "bold")).grid(row=0, column=0, sticky="w")
        self.badge = tk.Label(title, textvariable=self.badge_var, background="#edf2f4", foreground=MUTED,
                              font=(self.ui_font, 9), padx=11, pady=6)
        self.badge.grid(row=0, column=1, sticky="e")
        ttk.Label(results, textvariable=self.chain_var, style="Muted.TLabel", wraplength=740,
                  font=(self.ui_font, 9)).grid(row=1, column=0, sticky="w", pady=(5, 15))
        stats = ttk.Frame(results, style="Card.TFrame")
        stats.grid(row=2, column=0, sticky="ew", pady=(0, 18))
        for index, label in enumerate(["活动关节", "验证配置", "最大位置差 (m)", "最大旋转差 (rad)"]):
            stats.columnconfigure(index, weight=1)
            cell = tk.Frame(stats, background="#f3f7f8", padx=12, pady=10)
            cell.grid(row=0, column=index, sticky="ew", padx=(0, 8 if index < 3 else 0))
            tk.Label(cell, text=label, background="#f3f7f8", foreground=MUTED,
                     font=(self.ui_font, 9)).pack(anchor="w")
            tk.Label(cell, textvariable=self.stat_vars[index], background="#f3f7f8", foreground=INK,
                     font=(self.ui_font, 17, "bold")).pack(anchor="w", pady=(4, 0))
        self.notebook = ttk.Notebook(results)
        self.notebook.grid(row=3, column=0, sticky="nsew")
        table_frame = ttk.Frame(self.notebook, style="Card.TFrame")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)
        self.notebook.add(table_frame, text="DH 参数")
        columns = ("index", "joint", "type", "a", "alpha", "d", "theta")
        self.table = ttk.Treeview(table_frame, columns=columns, show="headings", height=10)
        for name, label, width, anchor in [
            ("index", "#", 34, "center"), ("joint", "关节", 160, "w"),
            ("type", "类型", 72, "center"), ("a", "a (m)", 110, "e"),
            ("alpha", "α (rad)", 110, "e"), ("d", "d₀ (m)", 110, "e"),
            ("theta", "θ₀ (rad)", 110, "e"),
        ]:
            self.table.heading(name, text=label, anchor=anchor)
            self.table.column(name, width=width, minwidth=width, anchor=anchor, stretch=name == "joint")
        self.table.tag_configure("even", background="#f4f8f9")
        self.table.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        horizontal = ttk.Scrollbar(table_frame, orient="horizontal", command=self.table.xview)
        horizontal.grid(row=1, column=0, sticky="ew")
        self.table.configure(yscrollcommand=scrollbar.set, xscrollcommand=horizontal.set)
        self.empty_label = tk.Label(table_frame, text="等待导入模型\n\n选择一个 URDF 文件和 DH / MDH 约定，开始提取。",
                                    foreground=MUTED, background=SURFACE,
                                    font=(self.ui_font, 11), justify="center")
        self.empty_label.place(relx=.5, rely=.48, anchor="center")
        self.matrix_text = self._text_tab("坐标变换")
        self.mapping_text = self._text_tab("变量映射")
        self.json_text = self._text_tab("完整 JSON")
        self.notes_text = self._text_tab("提示")
        footer = ttk.Frame(results, style="Card.TFrame")
        footer.grid(row=4, column=0, sticky="ew", pady=(15, 0))
        footer.columnconfigure(0, weight=1)
        self.copy_button = ttk.Button(footer, text="复制 JSON", command=self.copy_json)
        self.copy_button.grid(row=0, column=0, sticky="w")
        self.export_buttons = {}
        for index, (kind, label) in enumerate([("json", "保存 JSON"), ("csv", "保存 CSV"), ("markdown", "保存 Markdown")], 1):
            button = ttk.Button(footer, text=label, command=lambda value=kind: self.save_as(value))
            button.grid(row=0, column=index, padx=(8, 0))
            self.export_buttons[kind] = button
        ttk.Label(self.root, textvariable=self.formula_var,
                  background=BACKGROUND, foreground=MUTED, font=(self.ui_font, 9)).grid(
            row=2, column=0, sticky="w", padx=24, pady=(2, 12))

    def _text_tab(self, name: str) -> tk.Text:
        container = ttk.Frame(self.notebook, style="Card.TFrame", padding=(8, 10))
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        text = tk.Text(container, wrap="none", borderwidth=0, highlightthickness=0,
                       background=SURFACE, foreground=INK, font=("DejaVu Sans Mono", 10),
                       padx=10, pady=10, state="disabled")
        text.grid(row=0, column=0, sticky="nsew")
        vertical = ttk.Scrollbar(container, orient="vertical", command=text.yview)
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal = ttk.Scrollbar(container, orient="horizontal", command=text.xview)
        horizontal.grid(row=1, column=0, sticky="ew")
        text.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.notebook.add(container, text=name)
        return text

    @staticmethod
    def _set_text(widget: tk.Text, value: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.configure(state="disabled")

    def _toggle_advanced(self) -> None:
        if self.advanced_var.get():
            self.advanced_frame.grid()
            self.root.update_idletasks()
            self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all"))
            self.sidebar_canvas.yview_moveto(1.0)
        else:
            self.advanced_frame.grid_remove()

    def _scroll_sidebar(self, event: Any) -> None:
        if str(event.widget).startswith(str(self.sidebar_canvas)):
            step = -1 if getattr(event, "num", 0) == 4 or getattr(event, "delta", 0) > 0 else 1
            self.sidebar_canvas.yview_scroll(step, "units")

    def _set_badge(self, text: str, state: str = "neutral") -> None:
        colors = {"neutral": ("#edf2f4", MUTED), "success": ("#e1f3ec", "#147251"),
                  "error": ("#fbece9", "#b14a35"), "busy": ("#e5f2f3", ACCENT)}
        background, foreground = colors[state]
        self.badge_var.set(text)
        self.badge.configure(background=background, foreground=foreground)

    def _clear_result(self) -> None:
        self.result = None
        self.table.delete(*self.table.get_children())
        for variable in self.stat_vars:
            variable.set("—")
        for widget in (self.matrix_text, self.mapping_text, self.json_text, self.notes_text):
            self._set_text(widget, "")
        self.empty_label.configure(text="尚无提取结果\n\n选择模型与运动链后，点击「提取并验证 DH」。")
        self.empty_label.place(relx=.5, rely=.48, anchor="center")
        self._refresh_actions()

    def _update_convention_display(self) -> None:
        modified = CONVENTION_LABELS.get(self.convention_var.get()) == "modified"
        title = "MDH（Craig）" if modified else "标准 DH"
        order = ("Rx(α) · Tx(a) · Rz(θ) · Tz(d)" if modified
                 else "Rz(θ) · Tz(d) · Tx(a) · Rx(α)")
        self.formula_var.set(f"{title}：{order}   ｜   控制项目式建系 · 首轴尺寸入表 · Base/Tool 独立")
        self.table.heading("a", text="a(i−1) (m)" if modified else "a(i) (m)")
        self.table.heading("alpha", text="α(i−1) (rad)" if modified else "α(i) (rad)")
        self.notebook.tab(0, text=f"{title} 参数")

    def _inputs_changed(self, *_args: Any) -> None:
        if self._suspend_changes or self._closed:
            return
        self._update_convention_display()
        self._generation += 1
        self._clear_result()
        if self.document is not None:
            self.chain_var.set(f"{self.base_var.get()}  →  {self.tip_var.get()}")
            self._set_badge("参数已更改")
            self.status_var.set("参数已更新，请重新提取。" if not self.busy else "参数已更改；正在结束上一任务。")

    def _refresh_actions(self) -> None:
        ready = self.document is not None
        can_run = ready and bool(self.base_var.get()) and bool(self.tip_var.get()) and not self.busy
        self.run_button.configure(state="normal" if can_run else "disabled")
        for button in (*self.export_buttons.values(), self.copy_button):
            button.configure(state="normal" if self.result is not None and not self.busy else "disabled")
        for combo in (self.base_combo, self.tip_combo):
            combo.configure(state="readonly" if ready else "disabled")

    def choose_file(self) -> None:
        selected = filedialog.askopenfilename(parent=self.root, title="选择 URDF 模型",
            filetypes=[("URDF 模型", "*.urdf"), ("XML 模型", "*.xml"), ("所有文件", "*")])
        if selected:
            self.load_file(selected)

    def load_demo(self) -> None:
        self.load_file(demo_path())

    def load_file(self, path: str | Path) -> bool:
        if self._closed:
            return False
        self._generation += 1
        self.document = None
        self._suspend_changes = True
        self.base_var.set("")
        self.tip_var.set("")
        self.base_combo.configure(values=())
        self.tip_combo.configure(values=())
        self._suspend_changes = False
        self._clear_result()
        self.path_var.set(Path(path).name)
        self.file_info_var.set("正在读取模型…")
        try:
            document = load_document(path)
        except Exception as error:
            self.file_info_var.set("未能读取这个文件，请重新选择。")
            self.status_var.set(f"读取失败：{error}")
            self.chain_var.set("请载入有效的 URDF 文件")
            self._set_badge("导入失败", "error")
            messagebox.showerror("URDF 导入失败", str(error), parent=self.root)
            return False
        self.document = document
        self._suspend_changes = True
        self.base_combo.configure(values=document.links)
        self.tip_combo.configure(values=document.links)
        self.base_var.set(document.base_link)
        self.tip_var.set(document.tip_link)
        self._suspend_changes = False
        self.path_var.set(document.path.name)
        self.file_info_var.set(f"{len(document.links)} 个 link  ·  {len(document.model.joints)} 个 joint")
        self.chain_var.set(f"{document.base_link}  →  {document.tip_link}")
        self._set_badge("模型已载入")
        self.status_var.set("已读取模型，可以选择运动链并提取。" if not self.busy else "模型已更换；等待上一任务结束。")
        self._refresh_actions()
        return True

    def run_conversion(self) -> None:
        if self._closed or self.busy or self.document is None:
            return
        self._generation += 1
        generation = self._generation
        self._clear_result()
        try:
            options = parse_options(self.samples_var.get(), self.seed_var.get(),
                                    self.position_tol_var.get(), self.rotation_tol_var.get(),
                                    self.parallel_tol_var.get())
            if self.convention_var.get() not in CONVENTION_LABELS:
                raise ValueError("请选择标准 DH 或 MDH（Craig）。")
            options["convention"] = CONVENTION_LABELS[self.convention_var.get()]
            base, tip = self.base_var.get(), self.tip_var.get()
            if base not in self.document.links or tip not in self.document.links:
                raise ValueError("请从列表中选择有效的 Base link 和 Tip link。")
        except (ValueError, TypeError) as error:
            self.status_var.set(f"参数无效：{error}")
            self._set_badge("参数有误", "error")
            messagebox.showerror("请检查提取参数", str(error), parent=self.root)
            return
        document = self.document
        self.busy = True
        self.status_var.set(f"正在提取{self.convention_var.get()}，并对照原 URDF 验证…")
        self._set_badge("提取与验证中", "busy")
        self.progress.start(12)
        self._refresh_actions()
        result_queue = self._queue

        def work() -> None:
            try:
                result = convert_document(document, base, tip, **options)
            except Exception as error:
                result_queue.put((generation, None, error))
            else:
                result_queue.put((generation, result, None))

        threading.Thread(target=work, daemon=True, name="urdf-dh-conversion").start()

    def _poll_worker(self) -> None:
        if self._closed:
            return
        try:
            generation, result, error = self._queue.get_nowait()
        except queue.Empty:
            pass
        else:
            self.busy = False
            self.progress.stop()
            if generation == self._generation:
                if error is not None:
                    self._clear_result()
                    self.status_var.set(f"提取失败：{error}")
                    self._set_badge("提取失败", "error")
                    messagebox.showerror("DH 提取失败", str(error), parent=self.root)
                else:
                    self.result = result
                    self._show_result(result)
            else:
                self.status_var.set("上一任务已结束；请按当前文件与参数重新提取。")
            self._refresh_actions()
        self._after_id = self.root.after(60, self._poll_worker)

    def _show_result(self, model: Any) -> None:
        kinds = {"revolute": "旋转", "continuous": "连续", "prismatic": "移动"}
        for index, row in enumerate(model.rows, 1):
            self.table.insert("", "end", values=(index, row.joint_name, kinds.get(row.joint_type, row.joint_type),
                f"{row.a:.6g}", f"{row.alpha:.6g}", f"{row.d_offset:.6g}", f"{row.theta_offset:.6g}"),
                tags=("even",) if index % 2 == 0 else ())
        if model.rows:
            self.empty_label.place_forget()
        else:
            self.empty_label.configure(text="这条运动链只有固定变换\n\nDH 表为 0 行；基座、工具变换和验证结果仍可导出。")
        validation = model.validation or {}
        values = [str(len(model.rows)), str(validation.get("samples", "—")),
                  f"{validation['max_position_error']:.2e}" if validation else "—",
                  f"{validation['max_rotation_error']:.2e}" if validation else "—"]
        for variable, value in zip(self.stat_vars, values):
            variable.set(value)
        payload = model.to_dict()
        matrix_lines = ["单行约定：" + payload["transform_order"],
                        "参数索引：" + payload["parameter_indexing"], "", "B  基座固定变换", ""]
        matrix_lines.extend("  ".join(f"{value: .10g}" for value in row) for row in model.base_transform)
        matrix_lines.extend(["", "C  工具固定变换", ""])
        matrix_lines.extend("  ".join(f"{value: .10g}" for value in row) for row in model.tool_transform)
        boundary = model.boundary_frames
        if boundary:
            matrix_lines.extend(["", "建系：控制项目式；URDF 关节 X 轴优先。",
                f"安装入口：{boundary['mount_link']}；法兰出口：{boundary['flange_link']}",
                "B = mount_transform × base_alignment",
                "C = tool_alignment × flange_to_tip", "", "安装 Base（请求起点 → 安装入口）", ""])
            matrix_lines.extend("  ".join(f"{value: .10g}" for value in row) for row in boundary["mount_transform"])
            matrix_lines.extend(["", "Tool（法兰出口 → 请求终点）", ""])
            matrix_lines.extend("  ".join(f"{value: .10g}" for value in row) for row in boundary["flange_to_tip"])
        matrix_lines.extend(["", "T_base_tip(q) = B × A_1(q) × … × A_n(q) × C",
                             "", "长度：米；角度：弧度。完整导出保留双精度数值。"])
        self._set_text(self.matrix_text, "\n".join(matrix_lines))
        mapping = ["独立输入顺序：" + ", ".join(model.input_names), "",
                   "q_DH = multiplier × q[source_name] + offset", "",
                   "旋转行：theta = theta_offset + q_DH", "移动行：d = d_offset + q_DH", ""]
        for row in model.rows:
            mapping.append(f"{row.joint_name}: {row.multiplier:.10g} × {row.source_name} + {row.offset:.10g}"
                           f"  [direction={row.direction}]")
        mapping.extend(["", "direction 仅记录遍历方向；反向轴已包含运动方向，不要重复反号。"])
        self._set_text(self.mapping_text, "\n".join(mapping))
        self._set_text(self.json_text, json.dumps(model.to_dict(), ensure_ascii=False, indent=2, allow_nan=False))
        notes = list(model.warnings) or ["没有额外的几何提示。"]
        notes.extend(["", "DH / MDH 建系通常不唯一，请同时保留 B、C 和关节映射。",
                      "FK 验证检查与输入 URDF 的一致性，不代表实机标定精度。"])
        notes.extend(["", "当前建系沿用控制项目方式：首轴轴向固定距离进入第一行，末轴固定链保留为 Tool。",
                      "参数保留原始 URDF 精度；与名义预设表可能有微小角度和尺寸差。",
                      "任意 URDF 边界若需轴向对齐，会在 boundary_frames 中单独记录。"])
        if model.convention == "modified":
            notes.extend(["", "Craig MDH 每行 a、alpha 属于前一段（i−1）；第一行取 a=alpha=0。",
                          "第一行 d 保留首轴轴向尺寸；末段固定几何包含在工具变换 C 中。"])
        self._set_text(self.notes_text, "\n\n".join(notes))
        self.chain_var.set(f"{model.base_link}  →  {model.tip_link}")
        title = "MDH" if model.convention == "modified" else "标准 DH"
        self._set_badge(f"{title} · FK 验证通过", "success")
        self.status_var.set("提取与验证完成。可查看结果或保存文件。")

    def save_as(self, kind: str) -> None:
        if self.result is None or self.busy or self._closed:
            return
        suffixes = {"json": ".json", "csv": ".csv", "markdown": ".md"}
        if kind not in suffixes:
            raise ValueError("Unsupported export format")
        suffix = suffixes[kind]
        source = self.document.path if self.document else None
        stem = source.stem if source else "robot"
        kind_name = "mdh" if self.result.convention == "modified" else "standard_dh"
        prefix = re.sub(r"[^\w.-]+", "_", f"{stem}_{self.result.base_link}_{self.result.tip_link}")[:130]
        name = f"{prefix}_{kind_name}"
        initial_dir = self._last_export_dir or (source.parent if source else Path.cwd())
        selected = filedialog.asksaveasfilename(parent=self.root, title="保存 DH 结果",
            initialdir=str(initial_dir), initialfile=name + suffix, defaultextension=suffix,
            filetypes=[(kind.upper() + " 文件", "*" + suffix), ("所有文件", "*")])
        if not selected:
            return
        try:
            save_result(self.result, selected, kind, source_path=source)
        except Exception as error:
            self.status_var.set(f"保存失败：{error}")
            messagebox.showerror("无法保存结果", str(error), parent=self.root)
            return
        self._last_export_dir = Path(selected).parent
        self.status_var.set(f"已保存：{selected}")

    def copy_json(self) -> None:
        if self.result is not None and not self.busy:
            self.root.clipboard_clear()
            self.root.clipboard_append(json.dumps(self.result.to_dict(), ensure_ascii=False, indent=2, allow_nan=False))
            self.status_var.set("完整 JSON 已复制到剪贴板。")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._generation += 1
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
        self.progress.stop()
        self.root.destroy()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="URDF 标准 DH / Craig MDH 桌面提取工具")
    parser.add_argument("file", nargs="?", type=Path, help="启动时载入的 URDF（可选）")
    parser.add_argument("--demo", action="store_true", help="载入内置示例")
    parser.add_argument("--base", help="初始 Base link（可选）")
    parser.add_argument("--tip", help="初始 Tip link（可选）")
    parser.add_argument("--convention", choices=("standard", "modified", "mdh"), default="standard")
    args = parser.parse_args(argv)
    if args.demo and args.file:
        parser.error("--demo 与文件路径不能同时使用")
    try:
        root = tk.Tk()
    except tk.TclError as error:
        print(f"无法启动桌面界面：{error}。请在有图形显示的环境运行 urdf-dh-gui。", file=sys.stderr)
        return 2
    app = DHApp(root)
    app.convention_var.set("标准 DH" if args.convention == "standard" else "MDH（Craig）")
    path = demo_path() if args.demo else args.file
    if path is not None and app.load_file(path):
        for value, variable in ((args.base, app.base_var), (args.tip, app.tip_var)):
            if value is not None:
                if value not in app.document.links:
                    messagebox.showerror("无效的 link", f"模型中没有 link：{value}", parent=root)
                else:
                    variable.set(value)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
