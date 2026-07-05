"""Layout fixes for the cramped vendored ``sharppy.viz`` bottom insets.

The upstream Theta-E (``thetae``) and SR-Wind (``srwinds``) insets that sit
beneath the hodograph draw their axis labels into *fixed-size* boxes (15-20 px
wide) while sizing the label font from the widget height
(``round(height * 0.0512)``). On a large / high-DPI display the font grows past
those boxes, so the text is clipped:

* Theta-E pressure labels (e.g. ``500``-``900``) are right-aligned in a 20 px
  box, so the leading digit is cut off on the left edge.
* Theta-E theta-e labels (e.g. ``300``-``350``) live in a 15 px box, so they are
  cut off along the bottom; in the live GUI they can also overlap each other
  when every 10 K tick fits individually but the inset is too narrow for all
  of their text boxes.
* The SR-Wind title is drawn in a fixed 45x35 px box, which clips the normal
  inset label font.
* The SR-Wind "Classic Supercell" annotation is drawn in a 50 px box anchored at
  the 40 kt line, which overflows the right edge on a narrow inset.

SHARPpy Reimagined never edits the pip-installed upstream package; like the
Qt6 enum shim in :mod:`sharpmod.viz._qt6_compat`, this restores correct
behaviour by patching the vendored classes at runtime. The overrides only widen
the label boxes and clamp the annotation inside the widget -- the coordinate
transforms, tick lines, colors, and frame borders are untouched, so the plots
render identically apart from no longer clipping their text.

:func:`apply` is idempotent and best-effort: if the upstream package is absent
(e.g. a test environment without SHARPpy) it is a silent no-op.
"""

from __future__ import annotations

__all__ = ["apply"]

_APPLIED = False


def apply() -> bool:
    """Install the inset label-layout fixes on the vendored insets.

    Returns ``True`` when the fixes are installed (or already active), ``False``
    when the upstream ``sharppy`` inset modules or PySide6 are unavailable.
    Idempotent: repeated calls are no-ops after the first success.
    """
    global _APPLIED
    if _APPLIED:
        return True

    try:
        from qtpy import QtCore, QtGui
        import sharppy.sharptab.utils as utils
        from sharppy.viz.thetae import backgroundThetae
        from sharppy.viz.srwinds import backgroundWinds
    except Exception:
        return False

    # Scoped Qt6 enums (valid regardless of the enum-flatten shim ordering).
    Align = QtCore.Qt.AlignmentFlag
    Pen = QtCore.Qt.PenStyle
    Text = QtCore.Qt.TextFlag
    DONT_CLIP = Text.TextDontClip

    # -- Theta-E inset ----------------------------------------------------- #

    def _thetae_draw_isobar(self, p, qp):
        """Pressure ticks with a wide, non-clipping, left-anchored label."""
        pen = QtGui.QPen(self.fg_color, 1, Pen.SolidLine)
        qp.setPen(pen)
        qp.setFont(self.label_font)
        y1 = self.pres_to_pix(p)
        offset = 5
        qp.drawLine(self.lpad, y1, self.lpad + offset, y1)
        qp.drawLine(self.brx + self.rpad - offset, y1, self.brx + self.rpad, y1)
        # Left-anchor just inside the frame and never clip, so the full value
        # is visible however large the DPI-scaled font is.
        qp.drawText(int(self.lpad + offset + 2), int(y1) - 20, 60, 40,
                    DONT_CLIP | Align.AlignVCenter | Align.AlignLeft,
                    utils.INT2STR(p))

    def _thetae_draw_thetae(self, t, qp):
        """Theta-E ticks with the label centered in the bottom margin band."""
        pen = QtGui.QPen(self.fg_color, 1, Pen.SolidLine)
        qp.setPen(pen)
        qp.setFont(self.label_font)
        x1 = self.theta_to_pix(t)
        offset = 5
        qp.drawLine(x1, 0, x1, 0 + offset)
        qp.drawLine(x1, self.bry + self.tpad - offset, x1, self.bry + self.rpad)
        # Drop the label into the empty bottom padding band, centered on the
        # tick and unclipped, so it is fully visible instead of cut off. Also
        # keep a running right edge for this background draw: a live GUI inset
        # can be narrower than the CLI image while still fitting each label
        # individually, which makes "300 310 320 ..." collide unless
        # intermediate labels are suppressed.
        if t <= 200:
            self._sharpmod_thetae_last_label_right = self.tlx - 9999
        label = utils.INT2STR(t)
        fm = QtGui.QFontMetrics(self.label_font)
        _adv = getattr(fm, "horizontalAdvance", None) or fm.width
        box_w = max(30, int(_adv(label)) + 8)
        box_h = max(16, int(fm.height()))
        left = int(round(float(x1) - box_w / 2.0))
        right = left + box_w
        last_right = getattr(
            self, "_sharpmod_thetae_last_label_right", self.tlx - 9999)
        gap = 3
        if left >= self.tlx and right <= self.brx and left >= last_right + gap:
            qp.drawText(left, int(self.bry) + 3, box_w, box_h,
                        DONT_CLIP | Align.AlignHCenter | Align.AlignTop,
                        label)
            self._sharpmod_thetae_last_label_right = right

    # -- SR-Wind inset ----------------------------------------------------- #

    def _winds_draw_height(self, h, qp):
        """Height ticks with a wide, non-clipping, left-anchored label."""
        pen = QtGui.QPen(self.fg_color, 1, Pen.SolidLine)
        qp.setPen(pen)
        qp.setFont(self.label_font)
        y1 = self.hgt_to_pix(h)
        offset = 5
        qp.drawLine(self.lpad, y1, self.lpad + offset, y1)
        qp.drawLine(self.brx + self.rpad - offset, y1, self.brx + self.rpad, y1)
        qp.drawText(int(self.lpad + offset + 2), int(y1) - 20, 40, 40,
                    DONT_CLIP | Align.AlignVCenter | Align.AlignLeft,
                    utils.INT2STR(h))

    def _winds_draw_frame(self, qp):
        """Frame + title identical to upstream, but the "SR Wind v. Height"
        title is drawn in the same title box as the adjacent Theta-E inset
        using the normal inset label font, and the Classic Supercell annotation
        is clamped inside the widget so it never spills off-edge."""
        pen = QtGui.QPen(self.fg_color, 2, Pen.SolidLine)
        qp.setPen(pen)
        # Upstream SR-Wind anchors this title at (15, 5, 45, 35), which makes
        # it sit higher and farther left than the adjacent Theta-E title at
        # (35, 15, 50, 50). Use the Theta-E title box and TextDontClip so the
        # normal label font keeps the same visual placement without clipping.
        title_font = QtGui.QFont(self.label_font)
        qp.setFont(title_font)
        qp.drawText(int(self.tlx) + 35, int(self.tly) + 15, 50, 50,
                    DONT_CLIP | Align.AlignVCenter | Align.AlignHCenter,
                    'SR Wind\nv.\nHeight')
        qp.setFont(self.label_font)
        ## frame borders (unchanged)
        qp.drawLine(self.tlx, self.tly, self.brx, self.tly)
        qp.drawLine(self.brx, self.tly, self.brx, self.bry)
        qp.drawLine(self.brx, self.bry, self.tlx, self.bry)
        qp.drawLine(self.tlx, self.bry, self.tlx, self.tly)
        pen = QtGui.QPen(self.fg_color, 1, Pen.DashLine)
        qp.setPen(pen)
        zero = self.speed_to_pix(15.)
        qp.drawLine(zero, self.bry, zero, self.tly)
        lower = self.hgt_to_pix(8.)
        upper = self.hgt_to_pix(16.)
        classic1 = self.speed_to_pix(40.)
        classic2 = self.speed_to_pix(70.)
        pen = QtGui.QPen(self.clsc_color, 1, Pen.DashLine)
        qp.setPen(pen)
        qp.drawLine(classic1, lower, classic1, upper)
        qp.drawLine(classic2, lower, classic2, upper)
        # Center the label over the 40-70 kt classic-supercell band, then clamp
        # the box so it stays fully inside the widget (upstream anchored a
        # 50 px box at the 40 kt line, overflowing the right edge).
        tw = 74
        cx = (classic1 + classic2) / 2.0
        hi = self.brx - tw - 1
        lo = self.tlx + 1
        tx = int(min(max(cx - tw / 2.0, lo), hi)) if hi > lo else int(lo)
        qp.drawText(tx, 2, tw, 40,
                    Align.AlignVCenter | Align.AlignHCenter,
                    'Classic\nSupercell')

    _install(backgroundThetae, "draw_isobar", _thetae_draw_isobar)
    _install(backgroundThetae, "draw_thetae", _thetae_draw_thetae)
    _install(backgroundWinds, "draw_height", _winds_draw_height)
    _install(backgroundWinds, "draw_frame", _winds_draw_frame)

    _APPLIED = True
    return True


def _install(cls, name, func) -> None:
    """Bind ``func`` as ``cls.name`` once, tagging it so re-runs are no-ops."""
    existing = cls.__dict__.get(name)
    if getattr(existing, "_sharpmod_shim", False):
        return
    func._sharpmod_shim = True
    try:
        setattr(cls, name, func)
    except (AttributeError, TypeError):
        pass
