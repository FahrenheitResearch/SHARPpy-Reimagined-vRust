"""Hover cursor readouts for the vendored ``sharppy.viz`` bottom insets.

The upstream Storm Slinky (``slinky``), Theta-E (``thetae``) and Storm-Relative
Wind vs. Height (``srwinds``) insets that sit beneath the hodograph are static:
unlike the Skew-T and hodograph they have no cursor readout, so there is no way
to read a value off them by hovering. This module adds a lightweight,
non-destructive **hover readout** to each:

* **Theta-E** -- the pressure at the cursor and the profile's theta-e (K) at
  that level.
* **SR-Wind vs. Height** -- the height (km AGL) at the cursor and the
  storm-relative wind speed (kt) at that height.
* **Storm Slinky** -- the height (km) of the nearest plotted trajectory ring.

Like :mod:`sharpmod.viz.inset_layout` and the Qt6 enum shim, SHARPpy Reimagined
never edits the pip-installed upstream package; the behaviour is added by
patching the vendored ``plot*`` classes at runtime. The readout is painted as a
transient overlay *directly onto the widget* (never onto the backing
``plotBitMap``), so it does not accumulate, does not appear in exports, and the
coordinate transforms / plotted data are left completely untouched.

:func:`apply` is idempotent and best-effort: if the upstream package or PySide6
is unavailable (e.g. a headless test env without SHARPpy) it is a silent no-op.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "apply",
    "thetae_readout_lines",
    "srwinds_readout_lines",
    "slinky_readout_lines",
]

_APPLIED = False

#: Pixel search radius for the Storm Slinky "nearest trajectory ring" readout.
_SLINKY_HIT_RADIUS = 16.0


# ---------------------------------------------------------------------------
# Value computation -- each returns a ``list[str]`` of readout lines, or None
# when the cursor is outside the frame / the data is unavailable. Written as
# module-level functions (taking the widget as ``self``) so the coordinate
# inversion + interpolation can be unit-tested without a live Qt widget.
# ---------------------------------------------------------------------------

def thetae_readout_lines(self, x, y):
    """Theta-E inset: pressure at the cursor + profile theta-e (K) there."""
    prof = getattr(self, "prof", None)
    if prof is None:
        return None
    if not (self.tlx <= x <= self.brx and self.tpad <= y <= self.bry):
        return None
    span = float(self.bry - self.tpad)
    if span <= 0:
        return None
    p = self.pmax - ((self.bry - y) / span) * (self.pmax - self.pmin)
    if not np.isfinite(p) or p <= 0:
        return None
    pres = np.ma.masked_invalid(np.ma.asarray(self.pres, dtype=float))
    thte = np.ma.masked_invalid(np.ma.asarray(self.thetae, dtype=float))
    mask = np.ma.getmaskarray(pres) | np.ma.getmaskarray(thte)
    pres = np.asarray(pres[~mask], dtype=float)
    thte = np.asarray(thte[~mask], dtype=float)
    if pres.size < 2:
        return None
    order = np.argsort(pres)  # np.interp needs ascending sample points
    thte_val = float(np.interp(p, pres[order], thte[order]))
    return ["%.0f hPa" % p, "\u03b8e %.0f K" % thte_val]


def srwinds_readout_lines(self, x, y):
    """SR-Wind inset: height (km AGL) at the cursor + SR wind speed (kt)."""
    prof = getattr(self, "prof", None)
    if prof is None:
        return None
    if not (self.tlx <= x <= self.brx and self.tpad <= y <= self.bry):
        return None
    span = float(self.bry - self.tpad)
    if span <= 0:
        return None
    h = ((self.bry - 2 - y) / span) * (self.hmax - self.hmin) - self.hmin
    if not np.isfinite(h):
        return None
    h = max(h, 0.0)
    try:
        sfc_h = float(prof.hght[prof.sfc])
    except Exception:
        return None
    sru = np.ma.masked_invalid(np.ma.asarray(self.sru, dtype=float))
    srv = np.ma.masked_invalid(np.ma.asarray(self.srv, dtype=float))
    hght = np.ma.masked_invalid(np.ma.asarray(prof.hght, dtype=float))
    mask = (np.ma.getmaskarray(sru) | np.ma.getmaskarray(srv)
            | np.ma.getmaskarray(hght))
    sru = np.asarray(sru[~mask], dtype=float)
    srv = np.asarray(srv[~mask], dtype=float)
    agl = np.asarray(hght[~mask], dtype=float) - sfc_h
    if agl.size < 2:
        return None
    spd = np.hypot(sru, srv)
    order = np.argsort(agl)
    spd_val = float(np.interp(h * 1000.0, agl[order], spd[order]))
    return ["%.1f km" % h, "%.0f kt" % spd_val]


def slinky_readout_lines(self, x, y):
    """Storm Slinky: height (km) of the nearest plotted trajectory ring."""
    if getattr(self, "prof", None) is None or getattr(self, "pcl", None) is None:
        return None
    traj = getattr(self, "slinky_traj", None)
    if traj is None or traj is np.ma.masked:
        return None
    try:
        from sharppy.sharptab import utils as sp_utils
        qc = sp_utils.QC
    except Exception:
        qc = lambda v: v is not None and np.isfinite(v)  # noqa: E731
    best_z = None
    best_d2 = _SLINKY_HIT_RADIUS ** 2
    for pt in traj:
        tx, ty, tz = pt[0], pt[1], pt[2]
        if not qc(tx) or not qc(ty):
            continue
        px, py = self.xy_to_pix(tx, ty)
        d2 = (px - x) ** 2 + (py - y) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_z = tz
    if best_z is None or not np.isfinite(best_z):
        return None
    return ["%.1f km" % (float(best_z) / 1000.0)]


def apply() -> bool:
    """Install hover cursor readouts on the vendored inset ``plot*`` widgets.

    Returns ``True`` when the readouts are installed (or already active),
    ``False`` when the upstream ``sharppy`` inset modules or PySide6 are
    unavailable. Idempotent: repeated calls are no-ops after the first success.
    """
    global _APPLIED
    if _APPLIED:
        return True

    try:
        from qtpy import QtGui
        from sharppy.viz.thetae import plotThetae
        from sharppy.viz.srwinds import plotWinds
        from sharppy.viz.slinky import plotSlinky
    except Exception:
        return False

    # -- transient overlay painting ---------------------------------------- #

    def _draw_readout_box(self, qp, cx, cy, lines):
        from qtpy import QtCore
        font = QtGui.QFont("Helvetica", 8)
        qp.setFont(font)
        fm = QtGui.QFontMetrics(font)
        line_h = fm.height()
        try:
            text_w = max(fm.horizontalAdvance(t) for t in lines)
        except AttributeError:  # very old Qt bindings
            text_w = max(fm.width(t) for t in lines)
        pad = 4
        box_w = text_w + 2 * pad
        box_h = line_h * len(lines) + 2 * pad

        w = self.width()
        hgt = self.height()
        # Prefer up-and-to-the-right of the cursor; flip/clamp to stay inside.
        bx = cx + 12
        by = cy - box_h - 6
        if bx + box_w > w:
            bx = cx - box_w - 12
        bx = max(0, min(bx, w - box_w))
        if by < 0:
            by = cy + 12
        by = max(0, min(by, hgt - box_h))

        rect = QtCore.QRectF(float(bx), float(by), float(box_w), float(box_h))
        qp.setPen(QtGui.QPen(QtGui.QColor(120, 150, 180, 220), 1))
        qp.setBrush(QtGui.QBrush(QtGui.QColor(16, 20, 28, 214)))
        qp.drawRoundedRect(rect, 3.0, 3.0)
        qp.setPen(QtGui.QPen(QtGui.QColor(232, 240, 250)))
        ty = by + pad + fm.ascent()
        for t in lines:
            qp.drawText(int(bx + pad), int(ty), t)
            ty += line_h

    # -- generic mouse / paint wiring installed on each plot class --------- #

    def _install_readout(cls, compute):
        if getattr(cls, "_sharpmod_readout", False):
            return
        orig_paint = cls.paintEvent

        def paintEvent(self, e):
            orig_paint(self, e)
            if not getattr(self, "_sharpmod_readout_ready", False):
                # Enable hover events (mouseMoveEvent otherwise only fires while
                # a button is held). Done lazily so no __init__ patch is needed.
                self.setMouseTracking(True)
                self._sharpmod_readout_ready = True
            cur = getattr(self, "_sharpmod_cursor", None)
            if cur is None:
                return
            try:
                lines = compute(self, cur[0], cur[1])
            except Exception:
                lines = None
            if not lines:
                return
            qp = QtGui.QPainter()
            qp.begin(self)
            try:
                qp.setRenderHint(qp.Antialiasing)
                qp.setRenderHint(qp.TextAntialiasing)
                _draw_readout_box(self, qp, cur[0], cur[1], lines)
            finally:
                qp.end()

        def mouseMoveEvent(self, e):
            try:
                pos = e.position()
                self._sharpmod_cursor = (pos.x(), pos.y())
            except AttributeError:  # older Qt event API
                self._sharpmod_cursor = (e.x(), e.y())
            self.update()

        def leaveEvent(self, e):
            self._sharpmod_cursor = None
            self.update()

        cls.paintEvent = paintEvent
        cls.mouseMoveEvent = mouseMoveEvent
        cls.leaveEvent = leaveEvent
        cls._sharpmod_readout = True

    _install_readout(plotThetae, thetae_readout_lines)
    _install_readout(plotWinds, srwinds_readout_lines)
    _install_readout(plotSlinky, slinky_readout_lines)

    _APPLIED = True
    return True
