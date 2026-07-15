"""Install Cargo's cdylib under an importable stable-ABI filename.

Cargo intentionally emits a generic dynamic-library name. Python extension
modules need a platform-specific suffix, so release and local builds run this
small normalization step before tests or PyInstaller.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import os
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
CRATE = ROOT / "native" / "sharpmod-native"
PACKAGE = ROOT / "sharpmod"


def _cargo_output(profile: str) -> Path:
    target = Path(os.environ.get("CARGO_TARGET_DIR", CRATE / "target"))
    if sys.platform == "win32":
        name = "sharpmod_native.dll"
    elif sys.platform == "darwin":
        name = "libsharpmod_native.dylib"
    else:
        name = "libsharpmod_native.so"
    return target / profile / name


def _installed_name() -> str:
    if sys.platform == "win32":
        return "sharpmod_native.pyd"
    return "sharpmod_native.abi3.so"


def install(profile: str = "release") -> Path:
    source = _cargo_output(profile)
    if not source.is_file():
        raise FileNotFoundError(
            f"native extension was not built: {source}; run cargo build first"
        )

    PACKAGE.mkdir(parents=True, exist_ok=True)
    destination = PACKAGE / _installed_name()
    for pattern in ("sharpmod_native*.pyd", "sharpmod_native*.so"):
        for stale in PACKAGE.glob(pattern):
            if stale != destination:
                stale.unlink()
    shutil.copy2(source, destination)

    suffixes = importlib.machinery.EXTENSION_SUFFIXES
    if not any(destination.name.endswith(suffix) for suffix in suffixes):
        raise RuntimeError(
            f"{destination.name!r} is not importable on this interpreter; "
            f"known suffixes: {suffixes!r}"
        )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="release", choices=("debug", "release"))
    args = parser.parse_args()
    print(install(args.profile))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
