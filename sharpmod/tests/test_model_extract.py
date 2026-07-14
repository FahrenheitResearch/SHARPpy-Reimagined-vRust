"""Tests for the generic forecast-model point extractor."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from sharpmod.io import decoder as decoder_mod
from sharpmod.tools import model_extract


@pytest.fixture(autouse=True)
def _force_python_backend_for_extractor_unit_tests(monkeypatch):
    """Keep mocked extractor tests isolated from bundled native binaries."""
    monkeypatch.setenv("SHARPMOD_MODEL_BACKEND", "python")


def test_candidate_runs_walk_back_across_model_cycles():
    runs = model_extract.candidate_run_times(
        "gfs", datetime(2026, 7, 13, 13, tzinfo=timezone.utc), max_cycles=4)

    assert [run.strftime("%Y-%m-%d %H") for run in runs] == [
        "2026-07-13 12", "2026-07-13 06", "2026-07-13 00",
        "2026-07-12 18",
    ]


def test_extract_falls_back_to_previous_available_cycle(monkeypatch, tmp_path):
    requested = datetime(2026, 7, 13, 12, tzinfo=timezone.utc)
    seen = []

    def fake_probe(model, run_time=None, **kwargs):
        return {"available": run_time.hour == 6}

    def fake_extract(model, lat, lon, run_time=None, out_path=None, **kwargs):
        seen.append(run_time)
        return out_path

    monkeypatch.setattr(model_extract, "probe", fake_probe)
    monkeypatch.setattr(model_extract, "extract", fake_extract)
    out = tmp_path / "sounding.npz"

    path, resolved = model_extract.extract_with_cycle_fallback(
        "gfs", 35.0, -97.0, run_time=requested, out_path=str(out),
        max_cycles=3)

    assert path == str(out)
    assert resolved == datetime(2026, 7, 13, 6, tzinfo=timezone.utc)
    assert seen == [resolved]


def test_extract_fallback_error_lists_checked_cycles(monkeypatch, tmp_path):
    requested = datetime(2026, 7, 13, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(
        model_extract, "probe",
        lambda *args, **kwargs: {"available": False, "error": "not found"})

    with pytest.raises(model_extract.RetrievalError) as caught:
        model_extract.extract_with_cycle_fallback(
            "gfs", 35.0, -97.0, run_time=requested,
            out_path=str(tmp_path / "sounding.npz"), max_cycles=2)

    message = str(caught.value)
    assert "2026-07-13 12Z" in message
    assert "2026-07-13 06Z" in message
    assert "not found" in message
from sharpmod.tests.era5_synth import make_era5_dataset


def _dataset():
    ds = make_era5_dataset(
        lats=[34.0, 35.0],
        lons=[260.0, 261.0],
        levels=[1000.0, 850.0, 700.0],
        times=[datetime(2026, 7, 8, 0, tzinfo=timezone.utc)],
        seed=20,
    )
    vo = np.full(ds["t"].shape, 8.0e-5, dtype=float)
    return ds.assign(vo=(ds["t"].dims, vo))


def test_model_aliases_resolve_to_supported_configs():
    assert model_extract.get_config("ecmwf").key == "ecmwf-ifs"
    assert model_extract.get_config("ifs").key == "ecmwf-ifs"
    assert model_extract.get_config("aifs").key == "ecmwf-aifs"
    assert model_extract.get_config("ecmwf-aifs").herbie_model == "aifs"
    assert model_extract.get_config("nam3").key == "nam-3km-conus"
    assert model_extract.get_config("rrfs").key == "rrfs-a"


def test_every_model_search_accepts_non_mandatory_pressure_levels():
    """Every supported model requests all published pressure levels."""
    for cfg in model_extract.available_models():
        sample = ":t:975:pl:" if cfg.key.startswith("ecmwf-") \
            else ":TMP:975 mb:"
        assert re.search(cfg.search, sample), cfg.key


def test_forecast_hours_are_model_specific():
    hrrr_off_hour = model_extract.forecast_hours("hrrr", cycle_hour=5)
    assert max(hrrr_off_hour) == 18

    gfs_hours = model_extract.forecast_hours("gfs")
    assert 120 in gfs_hours
    assert 123 in gfs_hours
    assert 121 not in gfs_hours
    assert max(gfs_hours) == 384

    cfs_hours = model_extract.forecast_hours("cfs")
    assert cfs_hours[:3] == (0, 6, 12)
    assert max(cfs_hours) == 384


def test_domain_helpers_reject_out_of_region_points():
    assert model_extract.point_in_domain("hrrr", 35.0, -97.0)
    assert not model_extract.point_in_domain("hrrr", 52.0, 10.0)
    assert model_extract.point_in_domain("gfs", 52.0, 10.0)

    with pytest.raises(model_extract.ParameterRangeError, match="outside"):
        model_extract.extract(
            "hrrr", 52.0, 10.0,
            run_time=datetime(2026, 7, 8, 0, tzinfo=timezone.utc),
            out_path="unused.npz",
            dataset=_dataset(),
        )


def test_model_extract_writes_loadable_npz(tmp_path, monkeypatch):
    """A supported model writes the shared point-sounding format."""
    ds = _dataset()

    def _fake_retrieve(config, run_dt, fxx, member=None, download_dir=None):
        return ds, SimpleNamespace(grib="memory://gfs")

    monkeypatch.setattr(model_extract, "_retrieve_dataset", _fake_retrieve)

    out_path = tmp_path / "gfs_point.npz"
    result = model_extract.extract(
        "gfs", 35.0, -99.0,
        run_time=datetime(2026, 7, 8, 2, tzinfo=timezone.utc),
        fxx=6,
        out_path=str(out_path),
        loc="test-point",
    )

    assert result == str(out_path)
    with np.load(out_path, allow_pickle=True) as npz:
        assert str(npz["model"]) == "GFS"
        assert str(npz["loc"]) == "test-point"
        assert int(npz["fxx"]) == 6
        value = float(np.asarray(
            npz["surface_relative_vorticity"]).reshape(-1)[0])
        assert value == pytest.approx(8.0e-5)

    prof_collection, loc = decoder_mod.load_npz(str(out_path))
    assert loc == "test-point"
    assert prof_collection.getMeta("model") == "GFS"
    assert prof_collection.getMeta("surface_relative_vorticity") == pytest.approx(
        8.0e-5)


def test_extract_forwards_isolated_download_directory(tmp_path, monkeypatch):
    seen = {}

    def _fake_retrieve(config, run_dt, fxx, member=None, download_dir=None):
        seen["download_dir"] = download_dir
        return _dataset(), SimpleNamespace(grib="memory://gfs")

    monkeypatch.setattr(model_extract, "_retrieve_dataset", _fake_retrieve)
    download_dir = tmp_path / "downloads"
    model_extract.extract(
        "gfs", 35.0, -99.0,
        run_time=datetime(2026, 7, 8, 0, tzinfo=timezone.utc),
        out_path=str(tmp_path / "point.npz"),
        download_dir=str(download_dir),
    )

    assert seen["download_dir"] == str(download_dir)


def test_retrieve_dataset_suppresses_herbie_download_output(tmp_path, monkeypatch):
    """Herbie must not print Unicode status glyphs in Windows worker consoles."""
    seen = {}
    dataset = _dataset()

    class FakeHerbie:
        grib = "memory://hrrr"

        def __init__(self, *args, **kwargs):
            seen["constructor"] = kwargs

        def xarray(self, search, **kwargs):
            seen["search"] = search
            seen["xarray"] = kwargs
            print("👨🏻‍🏭 Created directory")
            return dataset

    class Cp1252Stream:
        def write(self, text):
            text.encode("cp1252")
            return len(text)

        def flush(self):
            pass

    monkeypatch.setitem(sys.modules, "herbie", SimpleNamespace(Herbie=FakeHerbie))
    monkeypatch.setattr(sys, "stdout", Cp1252Stream())
    config = model_extract.get_config("hrrr")
    returned, _herbie = model_extract._retrieve_dataset(
        config,
        datetime(2026, 7, 11, 12, tzinfo=timezone.utc),
        0,
        download_dir=tmp_path,
    )

    assert returned is dataset
    assert seen["xarray"]["verbose"] is False
    assert seen["xarray"]["remove_grib"] is False
    assert seen["xarray"]["save_dir"] == str(tmp_path)


def test_render_mode_removes_fetched_data_but_keeps_png(tmp_path, monkeypatch):
    npz_path = tmp_path / "point.npz"
    json_path = tmp_path / "point.json"
    png_path = tmp_path / "point.png"
    seen = {}

    def _fake_extract(*args, out_path=None, download_dir=None, **kwargs):
        seen["download_dir"] = Path(download_dir)
        Path(download_dir, "gfs", "raw.grib2").parent.mkdir(parents=True)
        Path(download_dir, "gfs", "raw.grib2").write_bytes(b"grib")
        Path(out_path).write_bytes(b"npz")
        Path(out_path).with_suffix(".json").write_text("{}", encoding="utf-8")
        return out_path

    def _fake_render(npz, png):
        assert Path(npz).exists()
        Path(png).write_bytes(b"png")
        return png

    monkeypatch.setattr(model_extract, "extract", _fake_extract)
    import sharpmod.tools as tools_mod
    monkeypatch.setattr(tools_mod, "render_npz", _fake_render)

    result = model_extract.main([
        "gfs", "35", "-99", str(npz_path), "--render", str(png_path),
    ])

    assert result == 0
    assert png_path.read_bytes() == b"png"
    assert not npz_path.exists()
    assert not json_path.exists()
    assert not seen["download_dir"].exists()


def test_render_failure_still_removes_all_fetched_data(tmp_path, monkeypatch):
    npz_path = tmp_path / "failed.npz"
    download_dirs = []

    def _fake_extract(*args, out_path=None, download_dir=None, **kwargs):
        download_dir = Path(download_dir)
        download_dirs.append(download_dir)
        Path(out_path).write_bytes(b"npz")
        Path(out_path).with_suffix(".json").write_text("{}", encoding="utf-8")
        (download_dir / "raw.grib2").write_bytes(b"grib")
        return out_path

    def _failed_render(npz, png):
        raise RuntimeError("render failed")

    monkeypatch.setattr(model_extract, "extract", _fake_extract)
    import sharpmod.tools as tools_mod
    monkeypatch.setattr(tools_mod, "render_npz", _failed_render)

    with pytest.raises(RuntimeError, match="render failed"):
        model_extract.main([
            "gfs", "35", "-99", str(npz_path), "--render", "failed.png",
        ])

    assert not npz_path.exists()
    assert not npz_path.with_suffix(".json").exists()
    assert len(download_dirs) == 1
    assert not download_dirs[0].exists()


def test_unsupported_model_reports_reason():
    with pytest.raises(model_extract.RetrievalError, match="not enabled"):
        model_extract.get_config("ukmet")
