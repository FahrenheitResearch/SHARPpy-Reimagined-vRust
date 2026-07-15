"""Desktop GUI bootstrap and compatibility facade.

Implementation lives in responsibility-focused ``gui_*`` modules. This module
selects the supported Qt binding before any Qt import, then re-exports the
historical entrypoints so ``sharpmod.gui:main`` and existing integrations remain
stable. Qt itself selects the native platform plugin unless the caller provides
an explicit ``QT_QPA_PLATFORM`` override.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_API", "pyside6")

from sharpmod import (  # noqa: E402
    gui_common as _common,
    gui_maps as _maps,
    gui_picker as _picker,
    gui_sessions as _sessions,
    gui_settings as _settings,
    gui_viewer as _viewer,
    gui_workers as _workers,
)

_IMPLEMENTATION_MODULES = (
    _common,
    _settings,
    _workers,
    _maps,
    _sessions,
    _viewer,
    _picker,
)

# Compatibility includes the private names exercised by the existing test and
# extension surface. New code should import from the focused owner module.
for _module in _IMPLEMENTATION_MODULES:
    for _name, _value in vars(_module).items():
        if not _name.startswith("__"):
            globals().setdefault(_name, _value)

PickerWindow = _picker.PickerWindow
PointMapWidget = _maps.PointMapWidget
StationMapWidget = _maps.StationMapWidget
compose_interactive = _viewer.compose_interactive
main = _picker.main

__all__ = [
    "PickerWindow",
    "PointMapWidget",
    "StationMapWidget",
    "compose_interactive",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
