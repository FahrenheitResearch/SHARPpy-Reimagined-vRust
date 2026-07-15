"""Regressions for desktop startup and background-worker ownership."""

from __future__ import annotations

import os
import inspect
import subprocess
import sys
import importlib
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["QT_API"] = "pyside6"

from sharpmod.gui import (  # noqa: E402
    PickerWindow,
    SOUNDING_FILE_FILTER,
    _ensure_setup,
)


class _Worker:
    def __init__(self):
        self.deleted = False

    def deleteLater(self) -> None:  # noqa: N802 - mirrors Qt API
        self.deleted = True


class _PickerDouble:
    def __init__(self, field: str):
        self.worker = _Worker()
        setattr(self, field, self.worker)
        self._sender = self.worker
        self.busy_updates = []

    def sender(self):
        return self._sender

    def _set_busy(self, busy: bool) -> None:
        self.busy_updates.append(busy)

    def _set_model_busy(self, busy: bool) -> None:
        self.busy_updates.append(busy)


def test_uwyo_worker_is_cleared_and_deleted_on_finish():
    picker = _PickerDouble("_worker")

    PickerWindow._on_fetch_worker_finished(picker)

    assert picker._worker is None
    assert picker.busy_updates == [False]
    assert picker.worker.deleted is True


def test_model_worker_is_cleared_before_qt_deletes_it():
    picker = _PickerDouble("_model_worker")

    PickerWindow._on_model_fetch_finished(picker)

    assert picker._model_worker is None
    assert picker.busy_updates == [False]
    assert picker.worker.deleted is True


def test_file_worker_is_cleared_and_open_button_restored():
    class Button:
        def __init__(self):
            self.enabled = False
            self.text = ""

        def setEnabled(self, value):  # noqa: N802
            self.enabled = value

        def setText(self, value):  # noqa: N802
            self.text = value

    picker = _PickerDouble("_file_worker")
    picker._file_open_btn = Button()

    PickerWindow._on_file_worker_finished(picker)

    assert picker._file_worker is None
    assert picker._file_open_btn.enabled is True
    assert picker._file_open_btn.text == "Open Sounding"
    assert picker.worker.deleted is True


def test_gui_leaves_native_platform_selection_to_qt():
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.pop("QT_QPA_PLATFORM", None)
    env["QT_API"] = "pyqt5"
    code = (
        "import os; import sharpmod.gui; "
        "print(repr(os.environ.get('QT_QPA_PLATFORM'))); "
        "print(os.environ['QT_API'])"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=root, env=env,
        check=True, capture_output=True, text=True,
    )

    assert completed.stdout.splitlines()[-2:] == ["None", "pyside6"]


def test_file_filter_advertises_spc_and_station_named_files():
    for extension in ("*.spc", "*.SPC", "*.oax", "*.OAX"):
        assert extension in SOUNDING_FILE_FILTER


def test_drop_selects_the_open_file_widget():
    class Value:
        def __init__(self):
            self.value = None

        def setText(self, value):  # noqa: N802 - mirrors Qt API
            self.value = value

        def setCurrentWidget(self, value):  # noqa: N802 - mirrors Qt API
            self.value = value

    class Url:
        def toLocalFile(self):  # noqa: N802 - mirrors Qt API
            return "example.OAX"

    class MimeData:
        def urls(self):
            return [Url()]

    class Event:
        def mimeData(self):  # noqa: N802 - mirrors Qt API
            return MimeData()

    class Picker:
        _file_tab = object()
        _tabs = Value()
        _file_edit = Value()

        def _open_file(self, path):
            self.opened = path

    picker = Picker()
    PickerWindow.dropEvent(picker, Event())

    assert picker._tabs.value is picker._file_tab
    assert picker._file_edit.value == "example.OAX"
    assert picker.opened == "example.OAX"


def test_model_busy_state_does_not_install_global_wait_cursor():
    source = inspect.getsource(PickerWindow._set_model_busy)
    assert "setOverrideCursor" not in source
    assert "restoreOverrideCursor" not in source


def test_live_gui_installs_the_ordered_render_patch_registry(monkeypatch):
    from sharpmod import gui_viewer

    calls = []
    app = object()

    class Renderer:
        @staticmethod
        def install_font(value):
            calls.append(("font", value))

        @staticmethod
        def _apply_sars_match_color():
            calls.append(("sars", None))

        @staticmethod
        def install_render_patches():
            calls.append(("patches", None))

    monkeypatch.setattr(gui_viewer, "_render", lambda: Renderer)
    monkeypatch.setattr(gui_viewer, "_setup_done", False)

    _ensure_setup(app)

    assert calls == [
        ("font", app),
        ("sars", None),
        ("patches", None),
    ]


def test_in_place_model_refresh_updates_index_board(monkeypatch):
    from sharpmod.viz import SPCWindow as spc_window

    calls = []

    class Board:
        def setData(self, prof, derived):  # noqa: N802
            calls.append((prof, derived))

    prof = object()
    derived = object()
    sw = type("SPCWidget", (), {
        "default_prof": prof,
        "index_board": Board(),
    })()
    monkeypatch.setattr(spc_window, "_derived_profile", lambda value: derived)

    spc_window._refresh_mounted_products(sw)

    assert calls == [(prof, derived)]


def test_cached_model_hour_turns_map_click_into_debounced_load():
    class Spin:
        def __init__(self):
            self.value = None

        def setValue(self, value):  # noqa: N802
            self.value = value

    class Timer:
        def __init__(self):
            self.started = False
            self.interval = None

        def setInterval(self, interval):  # noqa: N802
            self.interval = interval

        def start(self):
            self.started = True

    class Picker:
        _model_syncing_point = False
        _model_lat = Spin()
        _model_lon = Spin()
        _model_worker = None
        _model_click_timer = Timer()

        def _model_update_fetch_state(self):
            self.updated = True

        def _model_selected_hour_is_cached(self):
            return True

    picker = Picker()
    PickerWindow._model_on_map_point(picker, 35.25, -97.50)

    assert picker._model_lat.value == 35.25
    assert picker._model_lon.value == -97.50
    assert picker._model_click_timer.started is True
    assert picker._model_click_timer.interval == 40


def test_deferred_spc_construction_realizes_layout_offscreen(monkeypatch):
    module = importlib.import_module("sharpmod.viz.SPCWindow")

    class FakeWindow:
        calls = []

        def __init__(self, **_kwargs):
            self.show()
            self.raise_()

        def show(self):
            self.calls.append("show")

        def raise_(self):
            self.calls.append("raise")

        def setAttribute(self, _attribute, enabled):  # noqa: N802
            self.calls.append(f"offscreen:{enabled}")

        def hide(self):
            self.calls.append("hide")

    original_show = FakeWindow.show
    original_raise = FakeWindow.raise_
    monkeypatch.setattr(module, "SPCWindow", FakeWindow)

    module._construct_spc_window(object(), object(), defer_show=True)

    assert FakeWindow.calls == [
        "offscreen:True", "show", "hide", "offscreen:False"]
    assert FakeWindow.show is original_show
    assert FakeWindow.raise_ is original_raise
