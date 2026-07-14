import numpy as np
import pytest

from sharpmod.sharptab import native


def test_potential_temperature_matches_legacy_equation():
    pres = np.array([1000.0, 925.0, 850.0, 700.0, 500.0])
    tmpc = np.array([25.0, 20.0, 15.0, 5.0, -15.0])
    expected = (tmpc + 273.15) * np.power(1000.0 / pres, 0.28571426)

    assert np.allclose(native.potential_temperature(pres, tmpc), expected,
                       rtol=0.0, atol=1e-10)


@pytest.mark.skipif(not native.available(), reason="optional sharprs extension")
def test_native_profile_validation_accepts_physical_column():
    profile = native.validate_profile(
        [1000, 900, 800, 700], [100, 1000, 2000, 3100],
        [25, 18, 10, 1], [20, 12, 4, -6],
        [180, 200, 220, 240], [10, 20, 30, 40])
    assert len(profile) == 4
