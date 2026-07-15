"""Frozen-app packaging contracts for live forecast-model support."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_release_installs_model_fetch_dependencies():
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )
    assert 'python -m pip install -e ".[render,era5]"' in workflow
    assert "--model-fetch-runtime-check" in workflow
    assert "Verify frozen single-file runtime" in workflow
    assert workflow.count("--model-fetch-runtime-check") >= 5
    for job in ("build-windows", "build-linux", "build-macos"):
        assert f"  {job}:" in workflow
    for helper in ("rw_ingest", "rw_sharpmod", "rw_ecape_analytic"):
        assert helper in workflow
    assert (
        "RUSTY_WEATHER_REV: d656e5bdc0330142d9295be480f9efcb1d08c095"
        in workflow
    )
    assert workflow.count("native/sharpmod-native/Cargo.toml") >= 3
    assert workflow.count("packaging/install_native_extension.py") >= 3
    assert workflow.count("sharpmod_native_probe") >= 3
    assert "SHARPPYRS_REV: 958bcd685b1e28b8fce0ab5c7b8daea3cdd993aa" in workflow
    assert "sharppyrs-ecape-el.patch" in workflow


def test_pyinstaller_bundles_model_fetch_runtime():
    spec = (ROOT / "packaging" / "sharpmod_gui.spec").read_text(
        encoding="utf-8"
    )
    collection_block = spec.split("a = Analysis", 1)[0]
    for package in (
        "xarray", "herbie", "cfgrib", "eccodes", "cdsapi", "numcodecs",
        "pyproj", "ecape", "ecape_parcel",
    ):
        assert f'"{package}"' in collection_block

    excludes_block = spec.split("excludes=", 1)[1].split("]", 1)[0]
    assert '"cfgrib"' not in excludes_block
    assert '"herbie"' not in excludes_block

    # The checkout lives inside a wrapper folder.  Analysis must use the
    # repository root resolved by the spec, not a relative parent directory,
    # or the editable ``sharpmod`` package is absent on other machines.
    assert "pathex=[_REPO]" in spec
    assert 'pathex=[".."]' not in spec
    assert 'APP_NAME = "SHARPpy-Reimagined-vRust"' in spec
    assert '"CFBundleShortVersionString": APP_VERSION' in spec
    assert 'hiddenimports.append("sharpmod.sharpmod_native")' in spec
    assert 'binaries.append((_extension, "sharpmod"))' in spec


def test_frozen_runtime_check_imports_cds_client():
    launcher = (ROOT / "packaging" / "sharpmod_gui_launcher.py").read_text(
        encoding="utf-8"
    )

    assert "import cdsapi" in launcher
    assert "import numcodecs" in launcher
    assert "import pyproj" in launcher
    assert "from logging.handlers import RotatingFileHandler" in launcher
    assert "logging_handlers=bool(RotatingFileHandler)" in launcher
    assert "gui_entrypoint=callable(gui_main)" in launcher
    assert "native_ecape_probe=native_ecape_probe is not None" in launcher
    assert "rusty_weather=rusty_weather_available" in launcher
    assert "from sharpmod import sharpmod_native" in launcher
    assert "sharpmod_native_probe=sharpmod_native_probe" in launcher
    assert "sharpmod_native_analysis_probe=True" in launcher
    assert "sharpmod_native_version=sharpmod_native.__version__" in launcher


def test_native_binary_license_notices_are_bundled_and_exact():
    native_notices = (
        ROOT / "sharpmod" / "resources" / "bin" /
        "NATIVE-ANALYSIS-LICENSES.txt"
    ).read_text(encoding="utf-8")
    rusty_notice = (
        ROOT / "sharpmod" / "resources" / "bin" /
        "RUSTY-WEATHER-LICENSE.txt"
    ).read_text(encoding="utf-8")

    for project, revision in {
        "sharppyrs": "958bcd685b1e28b8fce0ab5c7b8daea3cdd993aa",
        "sharprs": "1601674e8be0a07eaa48a50ddf6b2cedc035324f",
        "ecape-rs": "82922534c02a888e773c50463b5a49d535606276",
    }.items():
        assert project in native_notices
        assert revision in native_notices
    assert "BSD 3-Clause License" in native_notices
    assert native_notices.count("MIT License") == 2
    assert "sharprs (MIT)" in rusty_notice
    assert "sharprs (BSD-3-Clause)" not in rusty_notice
