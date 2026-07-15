"""Headless tests for the opt-in Rust/Python viewer comparison."""

from __future__ import annotations

import os
from datetime import datetime
import threading
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import numpy.ma as ma
import pytest
from qtpy.QtCore import QEventLoop
from qtpy.QtWidgets import QApplication, QMainWindow, QMenu

from sharpmod import gui_compare, gui_picker


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _reference_profiles():
    direct, parcel_fields, categorical = gui_compare._comparison_schema()
    python = SimpleNamespace()
    rust = {"direct": {}, "parcels": {}, "categorical": {}}
    for spec in direct:
        setattr(python, spec.name, 1.0)
        rust["direct"][spec.name] = 1.0
    for label, attr in gui_compare._PARCEL_NAMES.items():
        parcel = SimpleNamespace()
        rust["parcels"][label] = {}
        for field, _tolerance in parcel_fields:
            setattr(parcel, field, 100.0)
            rust["parcels"][label][field] = 100.0
        setattr(python, attr, parcel)
    for field in categorical:
        setattr(python, field, "NONE")
        rust["categorical"][field] = "NONE"
    return rust, python


def test_field_summary_uses_release_tolerances_and_missing_states():
    rust, python = _reference_profiles()
    # PWAT absolute tolerance is 0.05; exercise both sides of the gate.
    python.pwat = 1.0
    rust["direct"]["pwat"] = 1.04
    # CIN uses the energy tolerance and should fail at this distance.
    python.dcape = 100.0
    rust["direct"]["dcape"] = 125.0
    # Strict missing-state reporting is distinct from an ordinary difference.
    python.tei = ma.masked
    rust["direct"]["tei"] = None
    python.ship = ma.masked
    rust["direct"]["ship"] = 1.0
    python.precip_type = ma.masked
    rust["categorical"]["precip_type"] = None

    rows = {row.field: row for row in
            gui_compare.compare_profile_values(rust, python)}

    assert rows["pwat"].status == "PASS"
    assert rows["dcape"].status == "FAIL"
    assert rows["tei"].status == "MISSING (both)"
    assert rows["ship"].status == "FAIL (missing)"
    assert rows["precip_type"].status == "MISSING (both)"
    assert rows["dcape"].allowed == pytest.approx(10.0)


def test_profile_snapshot_is_detached_and_preserves_storm_motion():
    source = SimpleNamespace(
        pres=ma.array([1000.0, 900.0]),
        hght=ma.array([100.0, 1000.0]),
        tmpc=ma.array([20.0, 12.0]),
        dwpc=ma.array([15.0, 8.0]),
        wdir=ma.array([180.0, 200.0]),
        wspd=ma.array([10.0, 20.0]),
        omeg=ma.masked_array([-9999.0, -1.0], mask=[True, False]),
        latitude=0.0,
        location="TST",
        date=None,
        missing=-32768.0,
        srwind=(10.0, 20.0, -5.0, 8.0),
        bunkers=(9.0, 20.0, -5.0, 8.0),
    )

    snapshot, storm = gui_compare.snapshot_profile(source)
    source.tmpc[0] = 99.0

    assert snapshot["tmpc"][0] == 20.0
    assert snapshot["omeg"][0] == -32768.0
    assert snapshot["missing"] == -32768.0
    assert snapshot["latitude"] == 0.0
    assert storm == (10.0, 20.0, -5.0, 8.0)

    source.bunkers = source.srwind
    _snapshot, default_storm = gui_compare.snapshot_profile(source)
    assert default_storm is None


def test_profile_key_changes_after_an_in_place_trace_edit():
    profile = SimpleNamespace(
        pres=np.array([1000.0, 900.0]),
        hght=np.array([100.0, 1000.0]),
        tmpc=np.array([20.0, 12.0]),
        dwpc=np.array([15.0, 8.0]),
        wdir=np.array([180.0, 200.0]),
        wspd=np.array([10.0, 20.0]),
        omeg=np.array([-9999.0, -1.0]),
        latitude=35.0,
        missing=-9999.0,
        srwind=(10.0, 20.0, -5.0, 8.0),
    )
    collection = SimpleNamespace(
        getCurrentDate=lambda: datetime(2026, 7, 15, 12),
        getHighlightedMemberName=lambda: "deterministic",
    )

    before = gui_compare.RustPythonCompareController._profile_key(
        collection, profile)
    profile.tmpc[0] = 21.0
    after = gui_compare.RustPythonCompareController._profile_key(
        collection, profile)

    assert before != after


def test_trace_edit_while_worker_runs_makes_pending_result_stale(app):
    profile = SimpleNamespace(
        pres=np.array([1000.0, 900.0]),
        hght=np.array([100.0, 1000.0]),
        tmpc=np.array([20.0, 12.0]),
        dwpc=np.array([15.0, 8.0]),
        wdir=np.array([180.0, 200.0]),
        wspd=np.array([10.0, 20.0]),
        omeg=np.array([-9999.0, -1.0]),
        latitude=35.0,
        missing=-9999.0,
        srwind=(10.0, 20.0, -5.0, 8.0),
    )
    collection = SimpleNamespace(
        getCurrentDate=lambda: datetime(2026, 7, 15, 12),
        getHighlightedMemberName=lambda: "deterministic",
        getHighlightedProf=lambda: profile,
    )
    win = QMainWindow()
    win.spc_widget = SimpleNamespace(
        prof_collections=[collection], prof_ids=["TST Rust"], pc_idx=0)
    action = gui_compare.QAction("Comparing…", win)
    controller = gui_compare.RustPythonCompareController(win, action)
    controller._worker_token = 4
    key = controller._profile_key(collection, profile)

    assert controller._pending_is_current(4, key, collection, profile)
    profile.dwpc[0] = 14.0
    assert not controller._pending_is_current(4, key, collection, profile)
    win.close()


def test_watch_rows_use_deterministic_mmp_normalized_oracle(monkeypatch):
    from sharpmod.tools import native_parity

    rust, python = _reference_profiles()
    python.right_watch_type = "NONE"
    python.left_watch_type = "NONE"
    rust["categorical"]["right_watch_type"] = "TOR"
    rust["categorical"]["left_watch_type"] = "NONE"
    rust["normalization"] = {"mmp": 0.72}

    seen = []

    def normalized(legacy, native):
        seen.append((legacy, native.mmp))
        return {
            "right_watch_type": "TOR",
            "left_watch_type": "NONE",
        }, {"right_watch_type": ("NONE", "TOR")}

    monkeypatch.setattr(
        native_parity, "_deterministic_watch_oracle", normalized)

    rows = {row.field: row for row in
            gui_compare.compare_profile_values(rust, python)}

    assert seen == [(python, 0.72)]
    assert rows["right_watch_type"].python == "TOR"
    assert rows["right_watch_type"].status == "PASS (MMP-normalized)"
    assert rows["right_watch_type"].group == "categories: watch"
    assert rows["mmp"].rust == 0.72
    assert rows["mmp"].status == "INFO (legacy undefined)"


def test_legacy_builder_does_not_change_any_profile_target(monkeypatch):
    calls = []

    class Legacy:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def set_srright(self, u, v):
            calls.append(("right", u, v))

        def set_srleft(self, u, v):
            calls.append(("left", u, v))

        def get_watch(self):
            calls.append(("watch",))

    from sharppy.sharptab import profile as sp_profile
    monkeypatch.setattr(sp_profile, "ConvectiveProfile", Legacy)
    sentinel_target = object()
    collection = SimpleNamespace(_target_type=sentinel_target)

    built = gui_compare.build_legacy_profile(
        {"pres": np.array([1000.0]), "hght": np.array([100.0])},
        (1.0, 2.0, 3.0, 4.0),
    )

    assert isinstance(built, Legacy)
    assert collection._target_type is sentinel_target
    assert calls == [
        ("right", 1.0, 2.0), ("left", 3.0, 4.0), ("watch",)]


def test_install_adds_discoverable_action_without_starting_worker(app):
    win = QMainWindow()
    win.profilemenu = QMenu("Profiles", win)
    win.menuBar().addMenu(win.profilemenu)

    controller = gui_compare.install_profile_comparison(win)

    assert win._sharpmod_compare_controller is controller
    assert win._sharpmod_compare_action.text() == "Compare Rust vs Python…"
    assert controller._worker is None
    assert callable(win._sharpmod_prepare_profile_replacement)
    assert gui_compare.install_profile_comparison(win) is controller
    win.close()


def test_reference_calculation_runs_through_background_worker(
        app, monkeypatch):
    legacy = object()
    expected_rows = [
        gui_compare.CompareRow("test", "value", 1.0, 1.0, "PASS")]
    monkeypatch.setattr(
        gui_compare, "build_legacy_profile",
        lambda snapshot, storm: legacy)
    monkeypatch.setattr(
        gui_compare, "compare_profile_values",
        lambda rust, profile: expected_rows)
    worker = gui_compare._LegacyCompareWorker({}, None, {})
    result = []
    loop = QEventLoop()
    worker.succeeded.connect(
        lambda profile, rows: result.append((profile, rows)))
    worker.finished.connect(loop.quit)

    worker.start()
    loop.exec()
    worker.wait()

    assert result == [(legacy, expected_rows)]
    worker.deleteLater()


def test_detached_running_worker_is_retained_until_finish(app, monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    def slow_legacy(_snapshot, _storm):
        entered.set()
        assert release.wait(2.0)
        return object()

    monkeypatch.setattr(gui_compare, "build_legacy_profile", slow_legacy)
    monkeypatch.setattr(gui_compare, "compare_profile_values", lambda *_: [])
    win = QMainWindow()
    action = gui_compare.QAction("Comparing…", win)
    controller = gui_compare.RustPythonCompareController(win, action)
    worker = gui_compare._LegacyCompareWorker({}, None, {})
    controller._worker = worker
    loop = QEventLoop()
    finished = threading.Event()
    worker.finished.connect(finished.set)
    worker.finished.connect(loop.quit)

    worker.start()
    assert entered.wait(2.0)
    controller._detach_worker()
    assert worker in gui_compare._ORPHAN_WORKERS
    release.set()
    loop.exec()
    worker.wait()
    app.processEvents()

    assert worker not in gui_compare._ORPHAN_WORKERS
    assert finished.is_set()
    win.close()


def test_view_switch_uses_separate_collection_and_returns_in_place(app):
    from sharppy.sharptab import profile as sp_profile

    date = datetime(2026, 7, 15, 12)
    source_profile = SimpleNamespace(
        _sharpmod_calculation_backend="sharppyrs/sharprs")
    source = SimpleNamespace(
        _meta={
            "loc": "TST", "model": "HRRR", "run": date,
            "base_time": date, "observed": False,
        },
        getCurrentDate=lambda: date,
        getMeta=lambda name: {
            "loc": "TST", "model": "HRRR", "run": date,
            "base_time": date, "observed": False,
        }[name],
        getHighlightedProf=lambda: highlighted["profile"],
    )
    # Exact target type avoids ProfCollection's lazy conversion; no costly
    # analysis is needed to exercise the collection/view ownership seam.
    python_profile = object.__new__(sp_profile.ConvectiveProfile)
    python_profile.date = date
    highlighted = {"profile": source_profile}

    class SPCWidget:
        def __init__(self):
            self.prof_collections = [source]
            self.prof_ids = ["TST Rust"]
            self.pc_idx = 0

        def setProfileCollection(self, menu):  # noqa: N802
            self.pc_idx = self.prof_ids.index(menu)

    class Viewer(QMainWindow):
        def __init__(self):
            super().__init__()
            self.spc_widget = SPCWidget()

        @staticmethod
        def createMenuName(collection):  # noqa: N802
            return "TST Python"

        def addProfileCollection(self, collection, **_kwargs):  # noqa: N802
            self.spc_widget.prof_collections.append(collection)
            self.spc_widget.prof_ids.append("TST Python")
            self.spc_widget.pc_idx = 1

        def rmProfileCollection(self, menu):  # noqa: N802
            index = self.spc_widget.prof_ids.index(menu)
            self.spc_widget.prof_ids.pop(index)
            self.spc_widget.prof_collections.pop(index)
            self.spc_widget.pc_idx = 0

    win = Viewer()
    action = gui_compare.QAction("Compare Rust vs Python…", win)
    controller = gui_compare.RustPythonCompareController(win, action)
    entry = gui_compare._CompareEntry(
        (1,), source, "TST Rust", source_profile, python_profile, [])
    controller._cache[entry.key] = entry

    controller.show_python(entry)
    comparison = win.spc_widget.prof_collections[1]
    assert comparison is not source
    assert comparison.getHighlightedProf() is python_profile
    assert source._meta["model"] == "HRRR"
    assert win.spc_widget.pc_idx == 1

    replacement_profile = SimpleNamespace(
        _sharpmod_calculation_backend="sharppyrs/sharprs")
    highlighted["profile"] = replacement_profile
    assert controller._active_source()[1] is replacement_profile

    controller.show_rust(entry)
    assert win.spc_widget.pc_idx == 0
    assert win.spc_widget.prof_collections == [source]

    # The computed legacy object remains cached and can be remounted without
    # recalculating or reopening the sounding window.
    controller.show_python(entry)
    assert win.spc_widget.prof_collections[1].getHighlightedProf() \
        is python_profile
    controller.prepare_for_source_replacement()
    assert win.spc_widget.prof_collections == [source]
    assert entry.compare_collection is None
    win.close()


def test_model_replacement_invalidates_temporary_comparison_first():
    events = []
    old = SimpleNamespace(menu_name="old")
    new = SimpleNamespace(menu_name="new")

    class Viewer:
        _sharpmod_prepare_profile_replacement = staticmethod(
            lambda: events.append("prepare"))

        @staticmethod
        def createMenuName(profile):  # noqa: N802
            return profile.menu_name

        @staticmethod
        def addProfileCollection(profile, **_kwargs):  # noqa: N802
            events.append(("add", profile.menu_name))

        @staticmethod
        def rmProfileCollection(menu):  # noqa: N802
            events.append(("remove", menu))

    picker = SimpleNamespace(_model_viewer_menu="old")
    monkey = pytest.MonkeyPatch()
    monkey.setattr(
        gui_picker.PickerWindow, "_replace_model_profile_in_place",
        staticmethod(lambda *_args: False))
    monkey.setattr(
        gui_picker.PickerWindow, "_replace_same_menu_model_profile",
        staticmethod(lambda *_args: False))
    try:
        result = gui_picker.PickerWindow._replace_model_profile(
            picker, Viewer(), new)
    finally:
        monkey.undo()

    assert result == "new"
    assert events[0] == "prepare"
    assert old.menu_name == "old"


def test_stale_worker_finish_reenables_action(app):
    win = QMainWindow()
    action = gui_compare.QAction("Comparing…", win)
    action.setEnabled(False)
    controller = gui_compare.RustPythonCompareController(win, action)

    class Worker:
        deleted = False

        def deleteLater(self):  # noqa: N802
            self.deleted = True

    worker = Worker()
    controller._worker = worker
    controller._worker_token = 2
    controller._worker_finished(1, worker)

    assert action.isEnabled()
    assert action.text() == "Compare Rust vs Python…"
    assert worker.deleted
    win.close()
