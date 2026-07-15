"""On-demand Rust-versus-legacy-Python comparison for sounding viewers.

The normal viewer path never constructs a legacy ``ConvectiveProfile``.  This
module is installed as one menu action and does that expensive work only after
the user explicitly requests a comparison.  The reference calculation runs in
a ``QThread`` and the resulting profile is kept in a separate collection, so
the fetched/native collection and the application's global target type are
never changed.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.ma as ma

from qtpy.QtCore import QObject, QThread, Signal, Qt
from qtpy.QtGui import QAction, QColor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


_PARCEL_NAMES = {
    "surface": "sfcpcl",
    "forecast": "fcstpcl",
    "most unstable": "mupcl",
    "mixed layer": "mlpcl",
    "effective": "effpcl",
}

# These are the parcel values people most often inspect in the plotted tables.
# Their tolerances come from the complete release parity schema; limiting the
# interactive table to this subset keeps the dialog useful rather than showing
# hundreds of parcel trace samples.
_DISPLAY_PARCEL_FIELDS = (
    "bplus", "bminus", "lclhght", "lfchght", "elhght", "cap", "mplhght",
)

_CACHE_LIMIT = 8
_ORPHAN_WORKERS: set[QThread] = set()
_WATCH_FIELDS = frozenset(("right_watch_type", "left_watch_type"))


@dataclass(frozen=True)
class CompareRow:
    """One display-sized field comparison."""

    group: str
    field: str
    rust: Any
    python: Any
    status: str
    error: float | None = None
    allowed: float | None = None


@dataclass
class _CompareEntry:
    key: tuple
    source_collection: Any
    source_menu: str
    rust_profile: Any
    python_profile: Any
    rows: list[CompareRow]
    compare_collection: Any = None
    compare_menu: str | None = None


def _plain_value(value):
    """Return a small, thread-safe scalar/tuple representation."""
    if value is None or ma.is_masked(value):
        return None
    array = ma.asarray(value)
    if array.ndim == 0:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        return number if math.isfinite(number) and number > -9000.0 else None
    result = []
    for item in array.ravel():
        if ma.is_masked(item):
            result.append(None)
            continue
        try:
            number = float(item)
        except (TypeError, ValueError):
            result.append(str(item))
            continue
        result.append(
            number if math.isfinite(number) and number > -9000.0 else None)
    return tuple(result)


def snapshot_profile(prof) -> tuple[dict, tuple | None]:
    """Copy only physical profile inputs for a background Python analysis."""
    missing = getattr(prof, "missing", -9999.0)
    try:
        missing = float(missing)
    except (TypeError, ValueError):
        missing = -9999.0
    if not math.isfinite(missing):
        missing = -9999.0
    latitude = _plain_value(getattr(prof, "latitude", 35.0))
    if latitude is None:
        latitude = 35.0
    kwargs = {}
    for name in ("pres", "hght", "tmpc", "dwpc", "wdir", "wspd", "omeg"):
        raw = getattr(prof, name, None)
        if raw is None:
            continue
        kwargs[name] = np.asarray(
            ma.asarray(raw, dtype=float).filled(missing), dtype=float).copy()
    if "pres" not in kwargs:
        raise ValueError("The highlighted sounding has no pressure profile.")
    if "omeg" not in kwargs:
        kwargs["omeg"] = np.full_like(kwargs["pres"], -9999.0, dtype=float)
    kwargs.update(
        latitude=float(latitude),
        location=str(getattr(prof, "location", "Sounding") or "Sounding"),
        date=getattr(prof, "date", None),
        missing=float(missing),
        strictQC=False,
    )
    storm_motion = _plain_value(getattr(prof, "srwind", None))
    if not isinstance(storm_motion, tuple) or len(storm_motion) != 4 \
            or any(value is None for value in storm_motion):
        storm_motion = None
    # Leave an untouched/default Bunkers vector to each backend's own solver;
    # otherwise the comparison would force ``srwind`` to agree and conceal a
    # real motion-vector difference. If the user dragged a storm-motion point,
    # apply that common explicit input to Python so downstream values remain a
    # meaningful apples-to-apples comparison.
    bunkers = _plain_value(getattr(prof, "bunkers", None))
    if storm_motion is not None and storm_motion == bunkers:
        storm_motion = None
    return kwargs, storm_motion


def _profile_input_fingerprint(prof) -> bytes:
    """Hash mutable physical inputs so edited profiles cannot reuse stale data."""
    digest = hashlib.blake2b(digest_size=16)
    for name in ("pres", "hght", "tmpc", "dwpc", "wdir", "wspd", "omeg"):
        digest.update(name.encode("ascii"))
        raw = getattr(prof, name, None)
        if raw is None:
            digest.update(b"\0")
            continue
        array = ma.asarray(raw, dtype=float)
        values = np.ascontiguousarray(array.filled(np.nan), dtype="<f8")
        mask = np.ascontiguousarray(ma.getmaskarray(array), dtype=np.uint8)
        digest.update(repr(values.shape).encode("ascii"))
        digest.update(values.tobytes())
        digest.update(mask.tobytes())
    for name in ("latitude", "missing"):
        digest.update(name.encode("ascii"))
        digest.update(repr(_plain_value(getattr(prof, name, None))).encode("ascii"))
    return digest.digest()


def _comparison_schema():
    """Load the exact committed release-parity field tolerances lazily."""
    from sharpmod.tools.native_parity import (
        CATEGORICAL_FIELDS,
        PARCEL_FIELDS,
        SCALAR_FIELDS,
        VECTOR_FIELDS,
    )

    direct = tuple(SCALAR_FIELDS) + tuple(VECTOR_FIELDS)
    parcels = tuple(
        (field, PARCEL_FIELDS[field]) for field in _DISPLAY_PARCEL_FIELDS)
    return direct, parcels, tuple(CATEGORICAL_FIELDS)


def extract_rust_values(prof) -> dict:
    """Snapshot the Rust-backed public values selected by the parity schema."""
    direct, parcel_fields, categorical = _comparison_schema()
    values = {
        "direct": {
            spec.name: _plain_value(getattr(prof, spec.name, None))
            for spec in direct
        },
        "categorical": {
            name: _plain_value(getattr(prof, name, None))
            for name in categorical
        },
        # Legacy MMP reads uninitialized working-array cells. It is therefore
        # not itself release-gated, but the deterministic Rust value is needed
        # to re-run legacy watch classification without allocator-dependent
        # false differences.
        "normalization": {
            "mmp": _plain_value(getattr(prof, "mmp", None)),
        },
        "parcels": {},
    }
    for label, attr in _PARCEL_NAMES.items():
        parcel = getattr(prof, attr, None)
        values["parcels"][label] = {
            field: _plain_value(getattr(parcel, field, None))
            for field, _tolerance in parcel_fields
        }
    return values


def _numeric_row(group, field, rust, python, tolerance) -> CompareRow:
    left = python if isinstance(python, tuple) else (python,)
    right = rust if isinstance(rust, tuple) else (rust,)
    if len(left) != len(right):
        return CompareRow(group, field, rust, python, "FAIL (shape)")

    if all(old is None and new is None for old, new in zip(left, right)):
        return CompareRow(group, field, rust, python, "MISSING (both)")
    if any((old is None) != (new is None) for old, new in zip(left, right)):
        return CompareRow(group, field, rust, python, "FAIL (missing)")

    comparisons = []
    for old, new in zip(left, right):
        if old is None and new is None:
            continue
        try:
            error = abs(float(new) - float(old))
            allowed = max(
                float(tolerance.absolute),
                float(tolerance.relative) * abs(float(old)),
            )
        except (TypeError, ValueError):
            return CompareRow(
                group, field, rust, python,
                "PASS" if new == old else "FAIL",
            )
        ratio = error / allowed if allowed > 0 else (0.0 if error == 0 else math.inf)
        comparisons.append((ratio, error, allowed))
    if not comparisons:
        return CompareRow(group, field, rust, python, "MISSING (both)")
    _ratio, error, allowed = max(comparisons, key=lambda item: item[0])
    return CompareRow(
        group, field, rust, python,
        "PASS" if error <= allowed else "FAIL",
        error, allowed,
    )


def compare_profile_values(rust_values: dict, python_profile) -> list[CompareRow]:
    """Compare public values using the release audit's field tolerances."""
    direct, parcel_fields, categorical = _comparison_schema()
    from sharpmod.tools.native_parity import _deterministic_watch_oracle

    rust_mmp = rust_values.get("normalization", {}).get("mmp")
    watch_oracle, _watch_seams = _deterministic_watch_oracle(
        python_profile, SimpleNamespace(mmp=rust_mmp))
    watch_normalized = rust_mmp is not None
    rows = []
    for spec in direct:
        rows.append(_numeric_row(
            spec.group,
            spec.name,
            rust_values["direct"].get(spec.name),
            _plain_value(getattr(python_profile, spec.name, None)),
            spec.tolerance,
        ))
    legacy_mmp = _plain_value(getattr(python_profile, "mmp", None))
    rows.append(CompareRow(
        "corrected legacy undefined",
        "mmp",
        rust_mmp,
        legacy_mmp,
        "MISSING (both)" if rust_mmp is None and legacy_mmp is None
        else "INFO (legacy undefined)",
    ))
    for label, attr in _PARCEL_NAMES.items():
        parcel = getattr(python_profile, attr, None)
        for field, tolerance in parcel_fields:
            rows.append(_numeric_row(
                f"parcel: {label}",
                field,
                rust_values["parcels"][label].get(field),
                _plain_value(getattr(parcel, field, None)),
                tolerance,
            ))
    for field in categorical:
        rust = _plain_value(rust_values["categorical"].get(field))
        python = _plain_value(
            watch_oracle.get(field, getattr(python_profile, field, None)))
        if rust is None and python is None:
            status = "MISSING (both)"
        else:
            status = "PASS" if rust == python else "FAIL"
        group = "categories"
        if watch_normalized and field in _WATCH_FIELDS:
            group = "categories: watch"
            status += " (MMP-normalized)"
        rows.append(CompareRow(group, field, rust, python, status))
    # Put actionable differences first, then stable group/field ordering.
    return sorted(
        rows,
        key=lambda row: (
            0 if row.status.startswith("FAIL") else
            2 if row.status.startswith("MISSING") else 1,
            row.group,
            row.field,
        ),
    )


def build_legacy_profile(snapshot: dict, storm_motion=None):
    """Construct an upstream Python profile without touching target globals."""
    from sharppy.sharptab import profile as sp_profile

    kwargs = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in snapshot.items()
    }
    legacy = sp_profile.ConvectiveProfile(**kwargs)
    if storm_motion is not None:
        legacy.set_srright(storm_motion[0], storm_motion[1])
        legacy.set_srleft(storm_motion[2], storm_motion[3])
        # Upstream setters update kinematics/severe fields but leave cached
        # watch labels stale. The temporary reference profile should remain a
        # coherent profile when the user chooses "Show Legacy Python".
        refresh_watch = getattr(legacy, "get_watch", None)
        if callable(refresh_watch):
            refresh_watch()
    legacy._sharpmod_calculation_backend = "legacy-python-reference"
    return legacy


class _LegacyCompareWorker(QThread):
    succeeded = Signal(object, object)
    failed = Signal(str)

    def __init__(self, snapshot, storm_motion, rust_values):
        # Deliberately no QObject parent: a viewer may close while Python is
        # finishing. The module-level orphan set retains the thread safely.
        super().__init__()
        self._snapshot = snapshot
        self._storm_motion = storm_motion
        self._rust_values = rust_values

    def run(self):
        try:
            legacy = build_legacy_profile(
                self._snapshot, self._storm_motion)
            rows = compare_profile_values(self._rust_values, legacy)
        except Exception as exc:  # noqa: BLE001 - reported in the viewer
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.succeeded.emit(legacy, rows)


def _format_value(value) -> str:
    if value is None:
        return "--"
    if isinstance(value, tuple):
        return "(" + ", ".join(_format_value(item) for item in value) + ")"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:.0f}"
        if abs(value) >= 100:
            return f"{value:.1f}"
        return f"{value:.2f}"
    return str(value)


class _ComparisonDialog(QDialog):
    """Concise result table plus safe in-view backend switches."""

    def __init__(self, entry: _CompareEntry, controller, parent=None):
        super().__init__(parent)
        self._entry = entry
        self._controller = controller
        self.setWindowTitle("Compare Rust vs Legacy Python")
        self.resize(900, 620)

        layout = QVBoxLayout(self)
        failed = sum(row.status.startswith("FAIL") for row in entry.rows)
        missing = sum(row.status.startswith("MISSING") for row in entry.rows)
        informational = sum(
            row.status.startswith("INFO") for row in entry.rows)
        passed = len(entry.rows) - failed - missing - informational
        summary = QLabel(
            f"{passed} within release tolerances  •  {failed} differences  "
            f"•  {missing} missing in both"
            + (f"  •  {informational} informational" if informational else "")
            + "\n"
            "Python is an on-demand reference calculation; the normal viewer "
            "continues to use the cached Rust profile."
            + (
                "\nWatch rows marked MMP-normalized re-run legacy watch "
                "logic with deterministic Rust MMP because upstream legacy "
                "MMP reads undefined working-array cells."
                if any("MMP-normalized" in row.status for row in entry.rows)
                else ""
            ),
            self,
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)

        table = QTableWidget(len(entry.rows), 7, self)
        table.setHorizontalHeaderLabels((
            "Group", "Field", "Rust", "Legacy Python", "|Difference|",
            "Allowed", "Result",
        ))
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setAlternatingRowColors(True)
        for row_index, row in enumerate(entry.rows):
            values = (
                row.group,
                row.field,
                _format_value(row.rust),
                _format_value(row.python),
                _format_value(row.error),
                _format_value(row.allowed),
                row.status,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if row.status.startswith("FAIL"):
                    item.setForeground(QColor("#b42318"))
                elif row.status.startswith("PASS"):
                    item.setForeground(QColor("#16794b"))
                elif row.status.startswith("INFO"):
                    item.setForeground(QColor("#9a6700"))
                table.setItem(row_index, column, item)
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(table, 1)

        controls = QHBoxLayout()
        rust_button = QPushButton("Show Rust (fast)", self)
        python_button = QPushButton("Show Legacy Python", self)
        rust_button.clicked.connect(
            lambda: controller.show_rust(entry))
        python_button.clicked.connect(
            lambda: controller.show_python(entry))
        controls.addWidget(rust_button)
        controls.addWidget(python_button)
        controls.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.Close, parent=self)
        buttons.rejected.connect(self.close)
        controls.addWidget(buttons)
        layout.addLayout(controls)


class RustPythonCompareController(QObject):
    """Own one viewer's worker, cache, dialog, and temporary collection."""

    def __init__(self, win, action):
        super().__init__(win)
        self.win = win
        self.action = action
        self._cache: OrderedDict[tuple, _CompareEntry] = OrderedDict()
        self._worker = None
        self._worker_token = 0
        self._pending = None
        self._dialog = None
        self._mounted_entry = None
        win.destroyed.connect(lambda *_args: self._detach_worker())

    def _status(self, message, timeout=0):
        try:
            self.win.statusBar().showMessage(message, timeout)
        except (AttributeError, RuntimeError):
            pass

    def _active_source(self):
        sw = getattr(self.win, "spc_widget", None)
        collections = getattr(sw, "prof_collections", None)
        prof_ids = getattr(sw, "prof_ids", None)
        if not isinstance(collections, list) or not collections:
            return None
        try:
            index = int(sw.pc_idx)
            collection = collections[index]
        except (AttributeError, IndexError, TypeError, ValueError):
            return None

        for entry in self._cache.values():
            if collection is entry.compare_collection:
                if entry.source_collection not in collections:
                    return None
                source_index = collections.index(entry.source_collection)
                try:
                    source_profile = entry.source_collection.getHighlightedProf()
                except Exception:
                    source_profile = entry.rust_profile
                return (
                    entry.source_collection,
                    source_profile,
                    prof_ids[source_index],
                )
        try:
            profile = collection.getHighlightedProf()
            menu = prof_ids[index]
        except (AttributeError, IndexError, TypeError, ValueError):
            return None
        return collection, profile, menu

    @staticmethod
    def _profile_key(collection, profile):
        storm = _plain_value(getattr(profile, "srwind", None))
        try:
            date = collection.getCurrentDate()
        except Exception:
            date = getattr(profile, "date", None)
        try:
            member = collection.getHighlightedMemberName()
        except Exception:
            member = None
        return (
            id(collection), id(profile), date, member, storm,
            _profile_input_fingerprint(profile),
        )

    def open(self):
        source = self._active_source()
        if source is None:
            QMessageBox.information(
                self.win, "Compare Rust vs Python",
                "Select a sounding profile before starting the comparison.")
            return
        collection, profile, menu = source
        if getattr(profile, "_sharpmod_calculation_backend", None) \
                != "sharppyrs/sharprs":
            QMessageBox.information(
                self.win, "Compare Rust vs Python",
                "The highlighted profile is already using the Python fallback; "
                "there is no Rust result to compare for this sounding.")
            return

        try:
            key = self._profile_key(collection, profile)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(
                self.win, "Compare Rust vs Python",
                f"The highlighted profile could not be fingerprinted:\n{exc}")
            return
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            self._show_dialog(cached)
            self._status("Reused cached Rust/Python comparison", 3500)
            return
        if self._worker is not None and self._worker.isRunning():
            return

        try:
            snapshot, storm_motion = snapshot_profile(profile)
            rust_values = extract_rust_values(profile)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(
                self.win, "Compare Rust vs Python",
                f"The highlighted profile could not be prepared:\n{exc}")
            return

        self._worker_token += 1
        token = self._worker_token
        self._pending = (token, key, collection, profile, menu)
        worker = _LegacyCompareWorker(snapshot, storm_motion, rust_values)
        self._worker = worker
        worker.succeeded.connect(
            lambda legacy, rows, token=token:
            self._comparison_ready(token, legacy, rows))
        worker.failed.connect(
            lambda message, token=token: self._comparison_failed(token, message))
        worker.finished.connect(
            lambda token=token, worker=worker:
            self._worker_finished(token, worker))
        self.action.setEnabled(False)
        self.action.setText("Comparing Rust vs Python…")
        self._status("Calculating legacy Python reference in background…")
        worker.start()

    def _pending_is_current(self, token, key, collection, profile):
        if token != self._worker_token:
            return False
        source = self._active_source()
        if source is None:
            return False
        current_collection, current_profile, _menu = source
        if current_collection is not collection or current_profile is not profile:
            return False
        try:
            return self._profile_key(collection, profile) == key
        except Exception:
            return False

    def _comparison_ready(self, token, legacy, rows):
        pending = self._pending
        if pending is None or pending[0] != token:
            return
        _token, key, collection, profile, menu = pending
        if not self._pending_is_current(token, key, collection, profile):
            self._status(
                "Sounding changed; discarded the stale Python comparison", 4500)
            return
        entry = _CompareEntry(
            key, collection, menu, profile, legacy, list(rows))
        self._cache[key] = entry
        self._cache.move_to_end(key)
        while len(self._cache) > _CACHE_LIMIT:
            discard_key = next((
                candidate_key
                for candidate_key, candidate in self._cache.items()
                if candidate is not self._mounted_entry
            ), None)
            if discard_key is None:
                break
            old_entry = self._cache.pop(discard_key)
            old_entry.compare_collection = None
        self._show_dialog(entry)
        failed = sum(row.status.startswith("FAIL") for row in rows)
        self._status(
            f"Rust/Python comparison complete: {failed} differences", 5000)

    def _comparison_failed(self, token, message):
        if token != self._worker_token:
            return
        QMessageBox.warning(
            self.win, "Compare Rust vs Python",
            "The legacy Python reference calculation failed:\n" + message)
        self._status("Legacy Python comparison failed", 5000)

    def _worker_finished(self, token, worker):
        was_current_worker = self._worker is worker
        if was_current_worker:
            self._worker = None
        if token == self._worker_token:
            self._pending = None
        if was_current_worker:
            try:
                self.action.setEnabled(True)
                self.action.setText("Compare Rust vs Python…")
            except RuntimeError:
                pass
        worker.deleteLater()

    def _detach_worker(self):
        self._worker_token += 1
        worker = self._worker
        self._worker = None
        if worker is not None and worker.isRunning():
            _ORPHAN_WORKERS.add(worker)

            def _release():
                _ORPHAN_WORKERS.discard(worker)
                worker.deleteLater()

            worker.finished.connect(_release)

    def _show_dialog(self, entry):
        if self._dialog is not None:
            try:
                self._dialog.close()
            except RuntimeError:
                pass
        dialog = _ComparisonDialog(entry, self, self.win)
        self._dialog = dialog
        dialog.finished.connect(
            lambda *_args, dialog=dialog:
            setattr(self, "_dialog", None)
            if self._dialog is dialog else None)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _remove_mounted_comparison(self, *, focus_source=True):
        entry = self._mounted_entry
        if entry is None or entry.compare_collection is None:
            return
        sw = getattr(self.win, "spc_widget", None)
        collections = getattr(sw, "prof_collections", [])
        if entry.compare_collection not in collections:
            entry.compare_collection = None
            entry.compare_menu = None
            self._mounted_entry = None
            return
        if focus_source and entry.source_collection in collections:
            try:
                self.win.spc_widget.setProfileCollection(entry.source_menu)
            except Exception:
                pass
        menu = entry.compare_menu
        if menu is not None:
            try:
                self.win.rmProfileCollection(menu)
            except Exception:
                pass
        entry.compare_collection = None
        entry.compare_menu = None
        self._mounted_entry = None

    @staticmethod
    def _comparison_collection(entry):
        from sharppy.sharptab import prof_collection
        from sharppy.sharptab import profile as sp_profile

        source = entry.source_collection
        try:
            meta = dict(source._meta)
        except Exception:
            meta = {}
        try:
            date = source.getCurrentDate()
        except Exception:
            date = getattr(entry.python_profile, "date", None)
        if date is None:
            date = getattr(entry.python_profile, "date", None)
        try:
            source_model = source.getMeta("model")
        except Exception:
            source_model = "Sounding"
        meta.update(
            model=f"{source_model} — Legacy Python",
            run=meta.get("run", date) or date,
            base_time=meta.get("base_time", date) or date,
            observed=bool(meta.get("observed", False)),
            highlight="Legacy Python",
        )
        return prof_collection.ProfCollection(
            {"Legacy Python": [entry.python_profile]},
            [date],
            target_type=sp_profile.ConvectiveProfile,
            **meta,
        )

    def show_python(self, entry):
        sw = getattr(self.win, "spc_widget", None)
        collections = getattr(sw, "prof_collections", [])
        if entry.source_collection not in collections:
            QMessageBox.information(
                self.win, "Compare Rust vs Python",
                "The source sounding has changed. Run the comparison again.")
            return
        if self._mounted_entry is not entry:
            self._remove_mounted_comparison(focus_source=False)
        if entry.compare_collection is None \
                or entry.compare_collection not in collections:
            collection = self._comparison_collection(entry)
            self.win.addProfileCollection(
                collection, focus=True, check_integrity=False)
            entry.compare_collection = collection
            entry.compare_menu = self.win.createMenuName(collection)
            self._mounted_entry = entry
        else:
            sw.setProfileCollection(entry.compare_menu)
        self._status("Showing legacy Python reference profile", 3500)

    def show_rust(self, entry):
        sw = getattr(self.win, "spc_widget", None)
        collections = getattr(sw, "prof_collections", [])
        if entry.source_collection not in collections:
            QMessageBox.information(
                self.win, "Compare Rust vs Python",
                "The source sounding is no longer in this viewer.")
            return
        if self._mounted_entry is entry:
            # Return to the source and remove the temporary collection. The
            # Python profile/summary stay cached, so Show Legacy Python can
            # remount it instantly without leaving a stale Profiles entry.
            self._remove_mounted_comparison(focus_source=True)
        else:
            sw.setProfileCollection(entry.source_menu)
        self._status("Showing cached Rust profile", 3500)

    def prepare_for_source_replacement(self):
        """Invalidate comparison state before an in-place model map refresh."""
        self._worker_token += 1
        self._pending = None
        self._remove_mounted_comparison(focus_source=True)
        self._cache.clear()
        if self._dialog is not None:
            try:
                self._dialog.close()
            except RuntimeError:
                pass
        self._status("Rust/Python comparison reset for the new sounding", 2500)


def install_profile_comparison(win):
    """Install the opt-in comparison action; return its controller."""
    existing = getattr(win, "_sharpmod_compare_controller", None)
    if existing is not None:
        return existing
    action = QAction("Compare Rust vs Python…", win)
    action.setStatusTip(
        "Calculate a legacy Python reference on demand and compare fields")
    menu = getattr(win, "profilemenu", None)
    if menu is None:
        menu = win.menuBar().addMenu("Profiles")
    menu.addSeparator()
    menu.addAction(action)
    controller = RustPythonCompareController(win, action)
    action.triggered.connect(controller.open)
    win._sharpmod_compare_controller = controller
    win._sharpmod_compare_action = action
    # PickerWindow calls this just before replacing a cached model profile.
    win._sharpmod_prepare_profile_replacement = \
        controller.prepare_for_source_replacement
    return controller


__all__ = [
    "CompareRow",
    "RustPythonCompareController",
    "build_legacy_profile",
    "compare_profile_values",
    "extract_rust_values",
    "install_profile_comparison",
    "snapshot_profile",
]
