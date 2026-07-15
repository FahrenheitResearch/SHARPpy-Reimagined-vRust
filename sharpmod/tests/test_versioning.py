"""Package metadata and visible application labels share one version."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sharpmod
from sharpmod import gui, render


ROOT = Path(__file__).resolve().parents[2]


def test_every_runtime_surface_uses_package_version():
    from sharpmod._version import __version__ as package_version

    assert sharpmod.__version__ == package_version
    assert gui.APP_VERSION == package_version
    assert render.application_label() == (
        f"SHARPpy Reimagined vRust v{package_version}")


def test_pyproject_reads_the_version_attribute():
    document = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "version" not in document["project"]
    assert "version" in document["project"]["dynamic"]
    assert document["tool"]["setuptools"]["dynamic"]["version"] == {
        "attr": "sharpmod._version.__version__",
    }


def test_release_metadata_reads_the_canonical_version():
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )
    spec = (ROOT / "packaging" / "sharpmod_gui.spec").read_text(
        encoding="utf-8"
    )

    assert 'runpy.run_path("sharpmod/_version.py")' in workflow
    assert 'default: "v0.3.2"' in workflow
    assert "CFBundleShortVersionString\": APP_VERSION" in spec


def test_native_crate_uses_the_canonical_version():
    from sharpmod._version import __version__ as package_version

    cargo = tomllib.loads((
        ROOT / "native" / "sharpmod-native" / "Cargo.toml"
    ).read_text(encoding="utf-8"))
    assert cargo["package"]["version"] == package_version
