"""Real Tk integration checks; run under xvfb-run on a headless host."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import threading
import time

import numpy as np
import pytest

tk = pytest.importorskip("tkinter")

from urdf_dh import DHModel, URDFModel


@pytest.fixture
def robot_path(tmp_path: Path) -> Path:
    path = tmp_path / "机械臂 中文.urdf"
    path.write_text(
        """<robot name="gui_fixture">
          <link name="base"/><link name="arm"/><link name="tip"/>
          <joint name="joint1" type="continuous">
            <parent link="base"/><child link="arm"/>
            <origin xyz="0.1 0.2 0.3" rpy="0.2 -0.3 0.4"/>
            <axis xyz="0 0 1"/>
          </joint>
          <joint name="tool" type="fixed">
            <parent link="arm"/><child link="tip"/>
            <origin xyz="0.4 0.0 0.1" rpy="0.1 0.2 0.3"/>
          </joint>
        </robot>""",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def gui_app(monkeypatch: pytest.MonkeyPatch):
    try:
        root = tk.Tk()
    except tk.TclError as error:
        pytest.skip(f"Tk display unavailable; run with xvfb-run: {error}")
    root.withdraw()
    from urdf_dh import gui

    dialogs: list[tuple[str, tuple, dict]] = []
    for name in ("showerror", "showinfo", "showwarning"):
        monkeypatch.setattr(
            gui.messagebox, name,
            lambda *args, _name=name, **kwargs: dialogs.append((_name, args, kwargs)),
        )
    callbacks: list[BaseException] = []
    root.report_callback_exception = lambda kind, value, traceback: callbacks.append(value)
    app = gui.DHApp(root)
    root.update()
    app.samples_var.set("4")
    yield gui, root, app, dialogs, callbacks
    app.close()
    assert not callbacks, f"Tk callbacks raised: {callbacks!r}"


def _pump(root, predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        root.update()
        if time.monotonic() >= deadline:
            pytest.fail("Timed out waiting for GUI background work")
        time.sleep(0.005)
    root.update()


def _assert_no_result(app) -> None:
    assert app.result is None
    assert app.table.get_children() == ()
    assert all(button.instate(["disabled"]) for button in app.export_buttons.values())


def _convert(root, app) -> DHModel:
    app.run_conversion()
    _pump(root, lambda: not app.busy)
    assert app.result is not None
    return app.result


def test_file_dialog_button_loads_chain_and_conversion(gui_app, robot_path, monkeypatch):
    gui, root, app, dialogs, _ = gui_app
    requests = []

    def choose(**kwargs):
        requests.append(kwargs)
        return str(robot_path)

    monkeypatch.setattr(gui.filedialog, "askopenfilename", choose)
    app.file_button.invoke()
    root.update()
    assert requests
    assert app.document is not None
    assert app.base_var.get() == "base"
    assert app.tip_var.get() == "tip"
    _assert_no_result(app)
    app.run_button.invoke()
    _pump(root, lambda: not app.busy)
    assert app.result is not None
    assert len(app.result.rows) == len(app.table.get_children()) == 1
    assert all(not button.instate(["disabled"]) for button in app.export_buttons.values())
    assert not any(name == "showerror" for name, _, _ in dialogs)


def test_export_dialogs_json_roundtrip_csv_and_markdown(gui_app, robot_path, tmp_path, monkeypatch):
    gui, root, app, dialogs, _ = gui_app
    app.load_file(robot_path)
    model = _convert(root, app)
    outputs = {"json": tmp_path / "导出.json", "csv": tmp_path / "导出.csv", "markdown": tmp_path / "导出.md"}
    for kind, target in outputs.items():
        monkeypatch.setattr(gui.filedialog, "asksaveasfilename", lambda _target=target, **kwargs: str(_target))
        app.export_buttons[kind].invoke()
        assert target.exists()
    restored = DHModel.from_dict(json.loads(outputs["json"].read_text(encoding="utf-8")))
    source = URDFModel.from_file(robot_path).chain("base", "tip")
    for value in (-1.3, 0.0, 2.1):
        np.testing.assert_allclose(restored.fk([value]), source.fk([value]), atol=1e-12)
    assert restored.validation == model.validation
    with outputs["csv"].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["joint_name"] == "joint1"
    assert float(rows[0]["a_m"]) == pytest.approx(model.rows[0].a)
    report = outputs["markdown"].read_text(encoding="utf-8")
    assert "基座固定变换 B" in report and "工具固定变换 C" in report
    assert "验证结果" in report
    assert not any(name == "showerror" for name, _, _ in dialogs)


@pytest.mark.parametrize("name,value", [
    ("samples_var", "0"), ("samples_var", "bad"), ("seed_var", "-1"),
    ("position_tol_var", "nan"), ("rotation_tol_var", "0"),
    ("parallel_tol_var", "1"),
])
def test_invalid_options_clear_previous_result(gui_app, robot_path, name, value):
    _, root, app, dialogs, _ = gui_app
    app.load_file(robot_path)
    _convert(root, app)
    getattr(app, name).set(value)
    _assert_no_result(app)
    app.run_conversion()
    _pump(root, lambda: not app.busy)
    _assert_no_result(app)
    assert any(kind == "showerror" for kind, _, _ in dialogs)


def test_chain_change_clears_result_and_supports_reverse(gui_app, robot_path):
    _, root, app, _, _ = gui_app
    app.load_file(robot_path)
    _convert(root, app)
    app.base_var.set("tip")
    _assert_no_result(app)
    app.tip_var.set("base")
    model = _convert(root, app)
    assert model.base_link == "tip" and model.tip_link == "base"
    source = URDFModel.from_file(robot_path).chain("tip", "base")
    np.testing.assert_allclose(model.fk([0.6]), source.fk([0.6]), atol=1e-12)


def test_failed_new_file_load_does_not_retain_old_document(gui_app, robot_path, tmp_path):
    _, root, app, dialogs, _ = gui_app
    app.load_file(robot_path)
    _convert(root, app)
    broken = tmp_path / "broken.urdf"
    broken.write_text("<robot>", encoding="utf-8")
    app.load_file(broken)
    assert app.document is None
    assert app.base_var.get() == app.tip_var.get() == ""
    _assert_no_result(app)
    assert any(kind == "showerror" for kind, _, _ in dialogs)


def test_fixed_only_path_can_export_validated_model(gui_app, tmp_path):
    _, root, app, _, _ = gui_app
    path = tmp_path / "fixed.urdf"
    path.write_text("""<robot name="fixed"><link name="start"/><link name="end"/>
        <joint name="fixed" type="fixed"><parent link="start"/><child link="end"/>
        <origin xyz="0.2 0.3 0.4" rpy="0.1 0.2 0.3"/></joint></robot>""", encoding="utf-8")
    app.load_file(path)
    model = _convert(root, app)
    assert not model.rows
    assert app.table.get_children() == ()
    assert model.validation is not None
    assert all(not button.instate(["disabled"]) for button in app.export_buttons.values())
    np.testing.assert_allclose(model.fk(), URDFModel.from_file(path).chain("start", "end").fk(), atol=1e-12)


def test_export_cannot_overwrite_input(gui_app, robot_path, monkeypatch):
    gui, root, app, dialogs, _ = gui_app
    app.load_file(robot_path)
    _convert(root, app)
    original = robot_path.read_bytes()
    monkeypatch.setattr(gui.filedialog, "asksaveasfilename", lambda **kwargs: str(robot_path))
    app.save_as("json")
    assert robot_path.read_bytes() == original
    assert any(kind == "showerror" for kind, _, _ in dialogs)


def test_cancelled_file_and_export_dialogs_leave_current_result(gui_app, robot_path, monkeypatch):
    gui, root, app, dialogs, _ = gui_app
    app.load_file(robot_path)
    model = _convert(root, app)
    monkeypatch.setattr(gui.filedialog, "askopenfilename", lambda **kwargs: "")
    monkeypatch.setattr(gui.filedialog, "asksaveasfilename", lambda **kwargs: "")
    app.file_button.invoke()
    app.export_buttons["json"].invoke()
    assert app.result is model
    assert not dialogs


def test_unknown_link_rejected_without_old_result(gui_app, robot_path):
    _, root, app, dialogs, _ = gui_app
    app.load_file(robot_path)
    _convert(root, app)
    app.tip_var.set("does_not_exist")
    app.run_conversion()
    _pump(root, lambda: not app.busy)
    _assert_no_result(app)
    assert any(kind == "showerror" for kind, _, _ in dialogs)


def test_background_conversion_error_is_reported_and_not_exportable(gui_app, robot_path, monkeypatch):
    gui, root, app, dialogs, _ = gui_app
    app.load_file(robot_path)
    _convert(root, app)

    def fail(*args, **kwargs):
        raise ValueError("Independent FK validation failed")

    monkeypatch.setattr(gui, "convert_document", fail)
    app.run_conversion()
    _pump(root, lambda: not app.busy)
    _assert_no_result(app)
    assert any(kind == "showerror" and "Independent FK" in str(args) for kind, args, _ in dialogs)


def _block_conversion(gui, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = gui.convert_document

    def delayed(*args, **kwargs):
        entered.set()
        if not release.wait(4):
            raise TimeoutError("Test did not release background conversion")
        return original(*args, **kwargs)

    monkeypatch.setattr(gui, "convert_document", delayed)
    return entered, release


def test_late_result_cannot_overwrite_changed_chain(gui_app, robot_path, monkeypatch):
    gui, root, app, _, _ = gui_app
    app.load_file(robot_path)
    entered, release = _block_conversion(gui, monkeypatch)
    app.run_conversion()
    try:
        _pump(root, entered.is_set)
        app.tip_var.set("arm")
        assert app.busy
        _assert_no_result(app)
    finally:
        release.set()
    _pump(root, lambda: not app.busy)
    _assert_no_result(app)
    model = _convert(root, app)
    assert model.tip_link == "arm"


def test_late_result_cannot_overwrite_new_file(gui_app, robot_path, tmp_path, monkeypatch):
    gui, root, app, _, _ = gui_app
    app.load_file(robot_path)
    entered, release = _block_conversion(gui, monkeypatch)
    app.run_conversion()
    new_path = tmp_path / "new.urdf"
    new_path.write_text(robot_path.read_text(encoding="utf-8").replace('name="base"', 'name="new_base"').replace('link="base"', 'link="new_base"'), encoding="utf-8")
    try:
        _pump(root, entered.is_set)
        app.load_file(new_path)
        assert app.document is not None
        assert app.base_var.get() == "new_base"
        _assert_no_result(app)
    finally:
        release.set()
    _pump(root, lambda: not app.busy)
    _assert_no_result(app)
    model = _convert(root, app)
    assert model.base_link == "new_base"


def test_close_during_conversion_has_no_worker_tk_calls(gui_app, robot_path, monkeypatch):
    gui, root, app, _, _ = gui_app
    app.load_file(robot_path)
    entered, release = _block_conversion(gui, monkeypatch)
    main_thread = threading.get_ident()
    errors = []
    monkeypatch.setattr(threading, "excepthook", lambda args: errors.append(args.exc_value))

    def guard(original):
        def checked(*args, **kwargs):
            if threading.get_ident() != main_thread:
                error = AssertionError("Worker touched a Tk object")
                errors.append(error)
                raise error
            return original(*args, **kwargs)
        return checked

    for name in ("after", "after_cancel", "destroy"):
        monkeypatch.setattr(root, name, guard(getattr(root, name)))
    for name in ("base_var", "tip_var", "convention_var", "formula_var", "samples_var", "seed_var", "position_tol_var", "rotation_tol_var", "parallel_tol_var"):
        variable = getattr(app, name)
        monkeypatch.setattr(variable, "get", guard(variable.get))
        monkeypatch.setattr(variable, "set", guard(variable.set))
    previous = set(threading.enumerate())
    app.run_conversion()
    try:
        _pump(root, entered.is_set)
        workers = [thread for thread in threading.enumerate() if thread not in previous]
        assert workers
        app.close()
    finally:
        release.set()
    for thread in workers:
        thread.join(timeout=4)
        assert not thread.is_alive()
    assert not errors


def _assert_fully_visible(root, widget) -> None:
    assert widget.winfo_ismapped(), f"Unmapped widget: {widget}"
    assert widget.winfo_width() > 0 and widget.winfo_height() > 0
    left, top = widget.winfo_rootx(), widget.winfo_rooty()
    right = left + widget.winfo_width()
    bottom = top + widget.winfo_height()
    # Bounds within the root alone can miss clipping by an intermediate frame.
    # Check each ancestor as well, without assuming any sidebar implementation.
    ancestor = widget.master
    while ancestor is not None:
        ancestor_left, ancestor_top = ancestor.winfo_rootx(), ancestor.winfo_rooty()
        bounds = (ancestor_left, ancestor_top,
                  ancestor_left + ancestor.winfo_width(), ancestor_top + ancestor.winfo_height())
        assert left >= bounds[0] and top >= bounds[1], (widget, "starts outside", ancestor, bounds)
        assert right <= bounds[2] and bottom <= bounds[3], (widget, "clipped by", ancestor, bounds)
        if ancestor is root:
            break
        ancestor = ancestor.master


@pytest.mark.parametrize("width,height", [(1280, 840), (1060, 740)])
@pytest.mark.parametrize("advanced", [False, True], ids=["basic", "advanced"])
def test_action_buttons_remain_visible_with_long_path(gui_app, robot_path, tmp_path, width, height, advanced):
    _, root, app, _, _ = gui_app
    folder = tmp_path / "deeply_nested_directory_for_gui_visibility_regression"
    folder.mkdir()
    source = folder / "robot_with_a_long_descriptive_filename_for_layout_regression.urdf"
    source.write_bytes(robot_path.read_bytes())
    app.load_file(source)
    root.geometry(f"{width}x{height}+0+0")
    root.deiconify()
    app.advanced_var.set(advanced)
    app._toggle_advanced()
    _pump(root, lambda: root.winfo_width() == width and root.winfo_height() == height)
    _convert(root, app)
    root.update_idletasks()
    root.update()
    _assert_fully_visible(root, app.run_button)
    for button in app.export_buttons.values():
        _assert_fully_visible(root, button)


def test_convention_selection_invalidates_result_and_exports_mdh(gui_app, robot_path, tmp_path, monkeypatch):
    gui, root, app, dialogs, _ = gui_app
    app.load_file(robot_path)
    assert _convert(root, app).convention == "standard"
    app.convention_combo.current(1)
    root.update()
    _assert_no_result(app)
    assert "Rx(α) · Tx(a) · Rz(θ) · Tz(d)" in app.formula_var.get()
    assert "i−1" in app.table.heading("a", "text")
    result = _convert(root, app)
    assert result.convention == "modified"
    source = URDFModel.from_file(robot_path).chain("base", "tip")
    for q in (-.7, 0, 1.2):
        np.testing.assert_allclose(result.fk([q]), source.fk([q]), atol=1e-12, rtol=0)
    assert "a_(i-1)" in app.matrix_text.get("1.0", "end")
    assert "末段固定几何" in app.notes_text.get("1.0", "end")
    requests = []
    for kind, suffix in (("json", ".json"), ("csv", ".csv"), ("markdown", ".md")):
        target = tmp_path / ("mdh" + suffix)
        def choose(_target=target, **kwargs):
            requests.append(kwargs)
            return str(_target)
        monkeypatch.setattr(gui.filedialog, "asksaveasfilename", choose)
        app.save_as(kind)
        assert requests[-1]["initialfile"].endswith("_mdh" + suffix)
    assert json.loads((tmp_path / "mdh.json").read_text())["convention"] == "modified"
    with (tmp_path / "mdh.csv").open() as handle:
        assert next(csv.DictReader(handle))["convention"] == "modified"
    assert "MDH（Craig）" in (tmp_path / "mdh.md").read_text()
    app.convention_combo.current(0)
    _assert_no_result(app)
    assert _convert(root, app).convention == "standard"
    assert not dialogs


def test_late_standard_result_cannot_overwrite_mdh_selection(gui_app, robot_path, monkeypatch):
    gui, root, app, _, _ = gui_app
    app.load_file(robot_path)
    entered, release = _block_conversion(gui, monkeypatch)
    app.run_conversion()
    try:
        _pump(root, entered.is_set)
        app.convention_var.set("MDH（Craig）")
        _assert_no_result(app)
    finally:
        release.set()
    _pump(root, lambda: not app.busy)
    _assert_no_result(app)
    assert _convert(root, app).convention == "modified"


@pytest.mark.parametrize("width,height", [(1280, 840), (1060, 740)])
def test_convention_selector_is_visible_by_default(gui_app, width, height):
    _, root, app, _, _ = gui_app
    root.geometry(f"{width}x{height}+0+0")
    root.deiconify()
    _pump(root, lambda: root.winfo_width() == width and root.winfo_height() == height)
    _assert_fully_visible(root, app.convention_combo)


@pytest.mark.parametrize("label,convention", [("标准 DH", "standard"), ("MDH（Craig）", "modified")])
def test_controller_style_table_and_boundary_display(gui_app, tmp_path, label, convention):
    from .test_controller_frames import controller_robot
    _, root, app, dialogs, _ = gui_app
    path = controller_robot(tmp_path / "e1_controller.urdf", rounded=True)
    app.load_file(path)
    app.base_var.set("arm_base_link")
    app.tip_var.set("tcp1")
    app.convention_var.set(label)
    model = _convert(root, app)
    assert model.convention == convention
    assert model.frame_policy == "controller"
    first = app.table.item(app.table.get_children()[0], "values")
    assert float(first[5]) == pytest.approx(.1745)
    np.testing.assert_allclose(model.base_transform, np.eye(4), atol=1e-13, rtol=0)
    matrices = app.matrix_text.get("1.0", "end")
    assert "安装入口：arm_base_link；法兰出口：link7" in matrices
    assert "Tool（法兰出口 → 请求终点）" in matrices
    assert "首轴轴向尺寸入表" in app.notes_text.get("1.0", "end") or "首轴轴向固定距离进入第一行" in app.notes_text.get("1.0", "end")
    assert not dialogs
