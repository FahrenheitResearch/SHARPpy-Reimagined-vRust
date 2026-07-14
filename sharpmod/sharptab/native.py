"""Safe optional bridge to the ``sharprs`` native extension.

Only operations with a verified contract live here. Composite parameters are
deliberately excluded because the Rust and legacy SHARPpy implementations do
not yet use identical parcel/layer definitions.
"""

from __future__ import annotations

import numpy as np


def available():
    try:
        import sharprs  # noqa: F401
    except ImportError:
        return False
    return True


def validate_profile(pres, hght, tmpc, dwpc, wdir, wspd):
    """Validate and construct a native profile when ``sharprs`` is installed.

    Returns the native profile, or ``None`` when the optional extension is not
    installed. Validation errors intentionally propagate to the caller.
    """
    try:
        import sharprs
    except ImportError:
        return None
    columns = [np.asarray(value, dtype=float) for value in (
        pres, hght, tmpc, dwpc, wdir, wspd)]
    return sharprs.Profile(*[value.tolist() for value in columns])


def potential_temperature(pres, tmpc):
    """Potential temperature in kelvin, native when available.

    ``sharprs.thermo.theta`` is bit-for-bit compatible with the legacy
    SHARPpy equation for finite inputs. The fallback preserves that contract.
    """
    pres = np.asarray(pres, dtype=float)
    tmpc = np.asarray(tmpc, dtype=float)
    try:
        import sharprs
    except ImportError:
        return (tmpc + 273.15) * np.power(1000.0 / pres, 0.28571426)
    return np.asarray(sharprs.thermo.theta(pres, tmpc), dtype=float)
