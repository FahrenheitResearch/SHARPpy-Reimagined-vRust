"""Mounted-product refresh regressions."""

from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from sharpmod.viz import SPCWindow as spc_window


class _Recorder:
    def __init__(self, method):
        self.calls = []
        setattr(self, method, lambda *args: self.calls.append(args))


def test_refresh_mounted_products_uses_current_profile(monkeypatch):
    derived = object()
    monkeypatch.setattr(spc_window, "_derived_profile", lambda _prof: derived)
    board = _Recorder("setData")
    stream = _Recorder("setProf")
    redraws = []
    sound = SimpleNamespace(
        clearData=lambda: redraws.append("clear"),
        plotData=lambda: redraws.append("plot"),
        update=lambda: redraws.append("update"),
    )
    sw = SimpleNamespace(
        default_prof="focused",
        index_board=board,
        streamwiseness=stream,
        sound=sound,
    )

    spc_window._refresh_mounted_products(sw)

    assert board.calls == [("focused", derived)]
    assert stream.calls == [("focused",)]
    assert sound._sharpmod_derived_profile is derived
    assert redraws == ["clear", "plot", "update"]


def test_refresh_mounted_products_tolerates_unmounted_widgets(monkeypatch):
    monkeypatch.setattr(spc_window, "_derived_profile", lambda prof: prof)
    sw = SimpleNamespace(default_prof="focused")

    spc_window._refresh_mounted_products(sw)


def test_update_hook_stages_derived_before_single_skew_draw(monkeypatch):
    """Profile replacement must not redraw the complete Skew-T afterwards."""
    events = []
    prof = object()
    derived = object()

    class Collection:
        def getHighlightedProf(self):  # noqa: N802 - vendored API
            return prof

    class DummySPCWidget:
        def __init__(self):
            self.prof_collections = [Collection()]
            self.pc_idx = 0
            self.default_prof = None
            self.sound = SimpleNamespace(
                clearData=lambda: events.append("clear"),
                plotData=lambda: events.append("plot"),
                update=lambda: events.append("update"),
            )
            self.index_board = _Recorder("setData")
            self.streamwiseness = _Recorder("setProf")

        def updateProfs(self):  # noqa: N802 - vendored API
            # This represents the one Skew-T/parcel draw inside upstream's
            # update.  Current derived data must already be attached here.
            assert self.sound._sharpmod_derived_profile is derived
            events.append("vendored-draw")
            self.default_prof = prof

        def toggleVector(self, _deviant):  # noqa: N802
            return None

        def swapInset(self):  # noqa: N802
            return None

    monkeypatch.setattr(spc_window, "_VendoredSPCWidget", DummySPCWidget)
    monkeypatch.setattr(spc_window, "_derived_profile", lambda _prof: derived)

    spc_window._install_streamwiseness_hooks()
    widget = DummySPCWidget()
    widget.updateProfs()

    assert events == ["vendored-draw"]
    assert widget.index_board.calls == [(prof, derived)]
    assert widget.streamwiseness.calls == [(prof,)]


def test_toggle_hook_skips_only_repeated_deviant(monkeypatch):
    """Same-side refreshes are free while real left/right switches survive."""
    base_calls = []
    focus_calls = []

    class DummySPCWidget:
        def toggleVector(self, deviant):  # noqa: N802 - vendored API
            base_calls.append(deviant)
            self.setFocus()

        def setFocus(self):  # noqa: N802 - Qt API
            focus_calls.append(True)

        def swapInset(self):  # noqa: N802 - vendored API
            return None

        def updateProfs(self):  # noqa: N802 - vendored API
            return None

    chart = _Recorder("setDeviant")
    monkeypatch.setattr(spc_window, "_VendoredSPCWidget", DummySPCWidget)

    spc_window._install_streamwiseness_hooks()
    widget = DummySPCWidget()
    widget.streamwiseness = chart

    widget.toggleVector("right")
    widget.toggleVector("right")
    widget.toggleVector("left")
    widget.toggleVector("left")
    widget.toggleVector("right")

    assert base_calls == ["right", "left", "right"]
    assert chart.calls == [("right",), ("left",), ("right",)]
    # Preserve upstream's focus side effect even for the two no-op refreshes.
    assert len(focus_calls) == 5
