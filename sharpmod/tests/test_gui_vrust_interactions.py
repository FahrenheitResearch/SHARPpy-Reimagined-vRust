"""Regressions for the responsive vRust picker/viewer interactions."""

from __future__ import annotations

import inspect
import os
from datetime import datetime, timezone
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from qtpy.QtWidgets import QApplication

from sharpmod import gui_common, gui_picker, gui_viewer, gui_workers


class _Signal:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)


class _Timer:
    def __init__(self):
        self.started = 0
        self.stopped = 0

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1


class _Spin:
    def __init__(self):
        self.value = None

    def setValue(self, value):  # noqa: N802 - Qt-compatible test double
        self.value = value


class _Worker:
    def __init__(self):
        self.deleted = False

    def deleteLater(self):  # noqa: N802 - Qt-compatible test double
        self.deleted = True


class _Button:
    def __init__(self):
        self.enabled = False
        self.label = ""

    def setEnabled(self, enabled):  # noqa: N802
        self.enabled = bool(enabled)

    def setText(self, label):  # noqa: N802
        self.label = label


class _StatusBar:
    def __init__(self):
        self.messages = []

    def showMessage(self, message, *_args):  # noqa: N802
        self.messages.append(message)


class _Profile:
    def __init__(self, menu_name):
        self.menu_name = menu_name


class _Viewer:
    def __init__(self, collections=None):
        self.collections = list(collections or [])
        self.events = []
        self.destroyed = _Signal()
        self.visible = True

    def isVisible(self):  # noqa: N802
        return self.visible

    def createMenuName(self, profile):  # noqa: N802
        return profile.menu_name

    def addProfileCollection(self, profile, **kwargs):  # noqa: N802
        self.events.append(("add", profile.menu_name, kwargs))
        self.collections.append((profile.menu_name, profile))

    def rmProfileCollection(self, menu_name):  # noqa: N802
        self.events.append(("remove", menu_name))
        self.collections = [
            item for item in self.collections if item[0] != menu_name]

    def setWindowTitle(self, title):  # noqa: N802
        self.events.append(("title", title))

    def showNormal(self):  # noqa: N802
        self.events.append(("show",))

    def raise_(self):
        self.events.append(("raise",))

    def activateWindow(self):  # noqa: N802
        self.events.append(("activate",))


class _MenuAction:
    def __init__(self, text):
        self._text = text

    def text(self):
        return self._text


class _MenuHandle:
    def __init__(self, visible=True):
        self._visible = visible

    def isVisible(self):  # noqa: N802
        return self._visible


class _ProfileMenu:
    def __init__(self, title):
        self._title = title
        self._handle = _MenuHandle()
        self._actions = [_MenuAction("Focus"), _MenuAction("Remove")]

    def title(self):
        return self._title

    def setTitle(self, title):  # noqa: N802
        self._title = title

    def menuAction(self):  # noqa: N802
        return self._handle

    def actions(self):
        return self._actions


class _Mapper:
    def __init__(self):
        self.mappings = []

    def setMapping(self, action, value):  # noqa: N802
        self.mappings.append((action.text(), value))


def test_compose_interactive_requests_deferred_native_construction():
    source = inspect.getsource(gui_viewer.compose_interactive)
    assert "defer_show=True" in source
    assert "win.hide()" not in source


def test_worker_completion_clears_references_before_delete():
    uwyo = _Worker()
    uwyo_busy = []
    picker = SimpleNamespace(
        _worker=uwyo,
        sender=lambda: uwyo,
        _set_busy=lambda busy: uwyo_busy.append(busy),
    )

    gui_picker.PickerWindow._on_fetch_worker_finished(picker)

    assert picker._worker is None
    assert uwyo_busy == [False]
    assert uwyo.deleted

    local = _Worker()
    button = _Button()
    picker = SimpleNamespace(
        _file_worker=local,
        _file_open_btn=button,
        sender=lambda: local,
    )

    gui_picker.PickerWindow._on_file_worker_finished(picker)

    assert picker._file_worker is None
    assert button.enabled
    assert button.label == "Open Sounding"
    assert local.deleted


def test_busy_states_do_not_install_a_global_wait_cursor():
    for method in (
            gui_picker.PickerWindow._set_busy,
            gui_picker.PickerWindow._set_model_busy,
            gui_picker.PickerWindow._open_file):
        source = inspect.getsource(method)
        assert "setOverrideCursor" not in source
        assert "restoreOverrideCursor" not in source


def test_file_filter_and_drop_target_the_real_open_file_tab():
    for extension in ("*.spc", "*.SPC", "*.oax", "*.OAX"):
        assert extension in gui_common.SOUNDING_FILE_FILTER

    class Value:
        value = None

        def setText(self, value):  # noqa: N802
            self.value = value

        def setCurrentWidget(self, value):  # noqa: N802
            self.value = value

    url = SimpleNamespace(toLocalFile=lambda: "example.OAX")
    event = SimpleNamespace(
        mimeData=lambda: SimpleNamespace(urls=lambda: [url]))
    picker = SimpleNamespace(
        _file_tab=object(),
        _tabs=Value(),
        _file_edit=Value(),
        _open_file=lambda path: setattr(picker, "opened", path),
    )

    gui_picker.PickerWindow.dropEvent(picker, event)

    assert picker._tabs.value is picker._file_tab
    assert picker._file_edit.value == "example.OAX"
    assert picker.opened == "example.OAX"


def test_map_click_autofetches_only_after_a_model_viewer_exists():
    timer = _Timer()
    picker = SimpleNamespace(
        _model_syncing_point=False,
        _model_lat=_Spin(),
        _model_lon=_Spin(),
        _model_worker=None,
        _model_click_timer=timer,
        _model_update_fetch_state=lambda: None,
        _model_viewer_is_open=lambda: False,
    )

    gui_picker.PickerWindow._model_on_map_point(picker, 35.25, -97.5)
    assert timer.started == 0

    picker._model_viewer_is_open = lambda: True
    gui_picker.PickerWindow._model_on_map_point(picker, 35.5, -97.25)

    assert timer.started == 1
    assert picker._model_lat.value == 35.5
    assert picker._model_lon.value == -97.25


def test_double_click_cancels_debounce_and_fetches_once():
    timer = _Timer()
    calls = []
    picker = SimpleNamespace(
        _model_click_timer=timer,
        _model_fetch=lambda: calls.append("fetch"),
    )

    gui_picker.PickerWindow._model_activate_point(picker, 35.0, -97.0)

    assert timer.stopped == 1
    assert calls == ["fetch"]


def test_native_hour_cache_worker_reports_progress_and_result(
        monkeypatch, tmp_path, qt_app):
    from sharpmod.tools import rusty_weather

    run = datetime(2026, 7, 15, 0, tzinfo=timezone.utc)
    hour_path = tmp_path / "store" / "hrrr" / "20260715_00z" / "f004.rws"
    hour_path.parent.mkdir(parents=True)
    hour_path.write_bytes(b"cached")
    calls = []
    progress = []
    results = []

    monkeypatch.setattr(rusty_weather, "default_cache_root", lambda: tmp_path)
    monkeypatch.setattr(rusty_weather, "is_available", lambda model: True)

    def cache_hour(model, run_time, fxx, **kwargs):
        calls.append((model, run_time, fxx))
        kwargs["progress"]("Trying HRRR via native provider", 15)
        return hour_path

    monkeypatch.setattr(rusty_weather, "cache_hour", cache_hour)
    worker = gui_workers._NativeHourCacheWorker("hrrr", run, 4)
    worker.progress.connect(
        lambda stage, total: progress.append((stage, total)))
    worker.finished_ok.connect(
        lambda *result: results.append(result))

    worker.run()

    assert calls == [("hrrr", run, 4)]
    assert progress == [("downloading", 0), ("complete", 0)]
    assert results == [("hrrr", run, 4, os.fspath(hour_path))]


def test_native_hour_cache_worker_surfaces_provider_error(
        monkeypatch, tmp_path, qt_app):
    from sharpmod.tools import model_extract, rusty_weather

    run = datetime(2026, 7, 15, 0, tzinfo=timezone.utc)
    failures = []
    monkeypatch.setattr(rusty_weather, "default_cache_root", lambda: tmp_path)
    monkeypatch.setattr(rusty_weather, "is_available", lambda model: True)
    monkeypatch.setattr(
        rusty_weather, "cache_hour",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            model_extract.RetrievalError("provider unavailable")))
    worker = gui_workers._NativeHourCacheWorker("hrrr", run, 4)
    worker.failed.connect(failures.append)

    worker.run()

    assert failures == [
        "Could not cache this model hour: provider unavailable"]


def test_cache_hour_action_starts_worker_without_blocking(monkeypatch):
    workers = []

    class Worker:
        def __init__(self, model, run_time, fxx, parent=None):
            self.request = (model, run_time, fxx, parent)
            self.finished_ok = _Signal()
            self.failed = _Signal()
            self.cancelled = _Signal()
            self.progress = _Signal()
            self.finished = _Signal()
            self.started = False
            workers.append(self)

        def start(self):
            self.started = True

    run = datetime(2026, 7, 15, 0, tzinfo=timezone.utc)
    busy = []
    prefetch = []
    status = _StatusBar()
    cfg = SimpleNamespace(key="hrrr", label="HRRR")
    picker = SimpleNamespace(
        _model_click_timer=_Timer(),
        _model_worker=None,
        _model_cache_worker=None,
        _model_config=lambda: cfg,
        _model_is_busy=lambda: False,
        _model_cache_availability=lambda _cfg: (True, "ready"),
        _cancel_model_prefetch=lambda **kwargs: prefetch.append(kwargs),
        _model_run_time=lambda: run,
        _model_selected_fxx=lambda: 4,
        _set_model_busy=lambda value, operation="fetch":
            busy.append((value, operation)),
        statusBar=lambda: status,
        _on_model_hour_cache_ok=lambda *_args: None,
        _on_model_hour_cache_failed=lambda *_args: None,
        _on_model_hour_cache_cancelled=lambda *_args: None,
        _on_model_fetch_progress=lambda *_args: None,
        _on_model_hour_cache_finished=lambda *_args: None,
    )
    monkeypatch.setattr(gui_picker, "_NativeHourCacheWorker", Worker)

    gui_picker.PickerWindow._cache_model_hour(picker)

    assert len(workers) == 1
    assert workers[0].request == ("hrrr", run, 4, picker)
    assert workers[0].started is True
    assert picker._model_cache_worker is workers[0]
    assert busy == [(True, "cache")]
    assert prefetch == [{"wait": True}]
    assert status.messages[-1].startswith(
        "Caching HRRR 2026-07-15 00Z F004")


def test_cancel_control_targets_active_native_hour_cache():
    class Worker:
        interrupted = False

        def requestInterruption(self):  # noqa: N802
            self.interrupted = True

    worker = Worker()
    cancel = _Button()
    status = _StatusBar()
    picker = SimpleNamespace(
        _model_worker=None,
        _model_cache_worker=worker,
        _model_cancel_btn=cancel,
        statusBar=lambda: status,
    )

    gui_picker.PickerWindow._cancel_model_fetch(picker)

    assert worker.interrupted is True
    assert cancel.enabled is False
    assert "model-hour cache" in status.messages[-1]


def test_native_hour_cache_completion_releases_worker():
    worker = _Worker()
    busy = []
    picker = SimpleNamespace(
        _model_cache_worker=worker,
        sender=lambda: worker,
        _set_model_busy=lambda value: busy.append(value),
    )

    gui_picker.PickerWindow._on_model_hour_cache_finished(picker)

    assert picker._model_cache_worker is None
    assert busy == [False]
    assert worker.deleted is True


@pytest.mark.parametrize(
    ("old_name", "new_name", "expected_order"),
    [
        ("Old model", "New model", ["add", "remove"]),
        ("Same model", "Same model", ["remove", "add"]),
    ],
)
def test_model_replacement_preserves_other_viewer_collections(
        old_name, new_name, expected_order):
    observed = _Profile("Observed")
    old_model = _Profile(old_name)
    new_model = _Profile(new_name)
    viewer = _Viewer([
        (observed.menu_name, observed),
        (old_model.menu_name, old_model),
    ])
    picker = SimpleNamespace(_model_viewer_menu=old_name)

    result = gui_picker.PickerWindow._replace_model_profile(
        picker, viewer, new_model)

    assert result == new_name
    assert [name for name, _profile in viewer.collections] == [
        "Observed", new_name]
    assert [event[0] for event in viewer.events] == expected_order


def test_same_menu_replaces_the_sole_model_profile_in_place():
    old_model = _Profile("Same model")
    new_model = _Profile("Same model")
    sound = SimpleNamespace(prof_collections=[old_model])
    hodo = SimpleNamespace(prof_collections=[old_model])
    updates = []
    spc_widget = SimpleNamespace(
        prof_ids=["Same model"],
        prof_collections=[old_model],
        sound=sound,
        hodo=hodo,
        pc_idx=0,
        updateProfs=lambda: updates.append("updated"),
    )
    viewer = _Viewer([("Same model", old_model)])
    viewer.spc_widget = spc_widget
    picker = SimpleNamespace(_model_viewer_menu="Same model")

    gui_picker.PickerWindow._replace_model_profile(
        picker, viewer, new_model)

    assert spc_widget.prof_collections == [new_model]
    assert sound.prof_collections == [new_model]
    assert hodo.prof_collections == [new_model]
    assert spc_widget.pc_idx == 0
    assert updates == ["updated"]
    assert viewer.events == []


def test_changed_menu_replaces_model_profile_with_one_refresh():
    observed = _Profile("Observed")
    old_model = _Profile("Old model")
    new_model = _Profile("New model")
    sound = SimpleNamespace(prof_collections=[observed, old_model])
    hodo = SimpleNamespace(prof_collections=[observed, old_model])
    updates = []
    spc_widget = SimpleNamespace(
        prof_ids=["Observed", "Old model"],
        prof_collections=[observed, old_model],
        sound=sound,
        hodo=hodo,
        pc_idx=1,
        updateProfs=lambda: updates.append("updated"),
    )
    viewer = _Viewer([
        (observed.menu_name, observed),
        (old_model.menu_name, old_model),
    ])
    old_menu = _ProfileMenu("Old model")
    viewer.spc_widget = spc_widget
    viewer.menu_items = [_ProfileMenu("Observed"), old_menu]
    viewer.focus_mapper = _Mapper()
    viewer.remove_mapper = _Mapper()
    picker = SimpleNamespace(_model_viewer_menu="Old model")

    result = gui_picker.PickerWindow._replace_model_profile(
        picker, viewer, new_model)

    assert result == "New model"
    assert spc_widget.prof_ids == ["Observed", "New model"]
    assert spc_widget.prof_collections == [observed, new_model]
    assert sound.prof_collections == [observed, new_model]
    assert hodo.prof_collections == [observed, new_model]
    assert spc_widget.pc_idx == 1
    assert old_menu.title() == "New model"
    assert viewer.focus_mapper.mappings == [("Focus", "New model")]
    assert viewer.remove_mapper.mappings == [("Remove", "New model")]
    assert updates == ["updated"]
    assert viewer.events == []


def test_fetch_result_reuses_tracked_model_viewer(monkeypatch, tmp_path):
    observed = _Profile("Observed")
    old_model = _Profile("Old model")
    new_model = _Profile("New model")
    viewer = _Viewer([
        (observed.menu_name, observed),
        (old_model.menu_name, old_model),
    ])
    retained = []
    monkeypatch.setattr(
        gui_picker, "QApplication",
        SimpleNamespace(processEvents=lambda: None))
    monkeypatch.setattr(
        gui_picker, "_retain_model_data_until_close",
        lambda *args: retained.append(args))

    class Picker:
        _model_progress_total = 0
        _model_viewer = viewer
        _model_viewer_menu = "Old model"

        def __init__(self):
            self.status = _StatusBar()

        def sender(self):
            return object()

        def statusBar(self):  # noqa: N802
            return self.status

        def _on_model_fetch_progress(self, *_args):
            pass

        def _model_viewer_is_open(self):
            return True

        def _replace_model_profile(self, target, profile):
            return gui_picker.PickerWindow._replace_model_profile(
                self, target, profile)

        def _track_model_viewer(self, target, menu_name):
            return gui_picker.PickerWindow._track_model_viewer(
                self, target, menu_name)

        def _clear_model_viewer(self, viewer_id=None):
            return gui_picker.PickerWindow._clear_model_viewer(
                self, viewer_id)

        def _show_sounding(self, *_args, **_kwargs):
            raise AssertionError("a second viewer must not be composed")

    picker = Picker()
    path = tmp_path / "sounding.npz"
    path.write_bytes(b"prepared")

    gui_picker.PickerWindow._on_model_fetch_ok(
        picker,
        str(path),
        "HRRR",
        datetime(2026, 7, 15, 0),
        1,
        new_model,
        "HRRR point",
    )

    assert picker._model_viewer is viewer
    assert picker._model_viewer_menu == "New model"
    assert [name for name, _profile in viewer.collections] == [
        "Observed", "New model"]
    assert any(event[0] == "show" for event in viewer.events)
    assert retained and retained[0][0] is viewer


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def test_forecast_controls_scroll_instead_of_overlapping(qt_app):
    picker = gui_picker.PickerWindow()
    try:
        picker._catalog_timer.stop()
        picker._avail_timer.stop()
        picker._model_availability_timer.stop()
        picker.resize(1000, 720)
        picker._tabs.setCurrentIndex(2)
        picker.show()
        qt_app.processEvents()

        assert picker._model_cache_btn.text() == (
            "Cache This Hour for Fast Map Browsing")
        assert picker._model_rail_scroll.verticalScrollBar().maximum() > 0
        assert (picker._model_time_box.geometry().bottom()
                < picker._model_point_box.geometry().top())
    finally:
        picker._catalog_timer.stop()
        picker._avail_timer.stop()
        picker._model_availability_timer.stop()
        picker.close()
