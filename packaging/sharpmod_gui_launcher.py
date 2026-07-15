"""Frozen-app entry point for the SHARPpy Reimagined GUI.

PyInstaller freezes THIS module as the executable's entry script. It simply
delegates to :func:`sharpmod.gui.main`, but keeping a dedicated launcher (rather
than pointing PyInstaller at ``sharpmod/gui.py`` directly) gives the bundle a
stable, import-safe ``__main__`` that never runs as part of the package.
"""

from __future__ import annotations

import multiprocessing
import json
import subprocess
import sys
from pathlib import Path


def _model_fetch_runtime_check(output_path: str) -> int:
    """Verify lazy GRIB dependencies inside a frozen release bundle."""
    result = {
        "ok": False,
        "frozen": bool(getattr(sys, "frozen", False)),
    }
    try:
        from logging.handlers import RotatingFileHandler

        import cdsapi
        import cfgrib
        import ecape_parcel
        import eccodes
        import herbie
        import numcodecs
        import pyproj
        import xarray

        from sharpmod import sharpmod_native
        from sharpmod.sharptab import native_ecape
        from sharpmod.gui import main as gui_main
        from sharpmod.tools import model_extract
        from sharpmod.tools import rusty_weather

        native_ecape_available = native_ecape.available()
        sharpmod_native_probe = bool(sharpmod_native.runtime_check())
        sharpmod_native_info = dict(sharpmod_native.backend_info())
        if sharpmod_native_info.get("schema") != "sharpmod.native-analysis.v1":
            raise RuntimeError("bundled native analysis returned the wrong schema")
        if sharpmod_native.__version__ != "0.3.2":
            raise RuntimeError(
                "bundled native analysis version does not match the app")
        sharpmod_analysis_probe = sharpmod_native.analyze(
            [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200],
            [100, 800, 1500, 3000, 4200, 5600, 7200, 9200, 10500, 12000],
            [30, 24, 18, 6, -2, -12, -24, -40, -49, -58],
            [22, 18, 12, 0, -8, -18, -32, -45, -55, -65],
            [180, 185, 195, 210, 220, 230, 240, 250, 255, 260],
            [10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
            latitude=35.0,
        )
        if sharpmod_analysis_probe.get("schema") != "sharpmod.native-analysis.v1":
            raise RuntimeError("bundled native analysis failed its sounding probe")
        rusty_weather_available = rusty_weather.is_available("hrrr")
        configured_models = len(model_extract.available_models())
        if configured_models < 1:
            raise RuntimeError("no model-fetch backend is configured")
        native_ecape_probe = None
        if native_ecape_available:
            native_ecape_probe = native_ecape.analytic_ecape(
                [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200],
                [100, 800, 1500, 3000, 4200, 5600, 7200, 9200, 10500, 12000],
                [30, 24, 18, 6, -2, -12, -24, -40, -49, -58],
                [22, 18, 12, 0, -8, -18, -32, -45, -55, -65],
                [5, 8, 12, 18, 22, 28, 34, 40, 44, 48],
                [2, 5, 8, 12, 16, 20, 24, 28, 32, 35],
                timeout=30,
            )
            if native_ecape_probe is None:
                raise RuntimeError("bundled native ECAPE helper failed its protocol probe")
        if rusty_weather_available:
            for executable in rusty_weather.find_binaries():
                subprocess.run(
                    [executable, "--help"],
                    capture_output=True,
                    check=True,
                    timeout=30,
                )

        result.update(
            cdsapi=bool(cdsapi.Client),
            cfgrib=cfgrib.__version__,
            ecape_parcel=getattr(ecape_parcel, "__version__", "installed"),
            eccodes=eccodes.codes_get_api_version(),
            herbie=herbie.__version__,
            numcodecs=numcodecs.__version__,
            pyproj=pyproj.__version__,
            xarray=xarray.__version__,
            configured_models=configured_models,
            native_ecape=native_ecape_available,
            native_ecape_probe=native_ecape_probe is not None,
            sharpmod_native=True,
            sharpmod_native_probe=sharpmod_native_probe,
            sharpmod_native_analysis_probe=True,
            sharpmod_native_version=sharpmod_native.__version__,
            sharpmod_native_engine=sharpmod_native_info.get("sharppyrs_revision"),
            rusty_weather=rusty_weather_available,
            ecape_fallback=not native_ecape_available,
            model_fetch_fallback=not rusty_weather_available,
            logging_handlers=bool(RotatingFileHandler),
            gui_entrypoint=callable(gui_main),
            ok=True,
        )
    except BaseException as exc:  # noqa: BLE001 - diagnostics must be recorded
        result["error"] = f"{type(exc).__name__}: {exc}"

    Path(output_path).write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0 if result["ok"] else 1


def _run() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--model-fetch-runtime-check":
        return _model_fetch_runtime_check(sys.argv[2])
    from sharpmod.gui import main
    return main(sys.argv)


if __name__ == "__main__":
    # Safe no-op when unfrozen; required so a bundled child process (some Qt /
    # scientific libs may spawn one) re-runs this launcher instead of the app.
    multiprocessing.freeze_support()
    raise SystemExit(_run())
