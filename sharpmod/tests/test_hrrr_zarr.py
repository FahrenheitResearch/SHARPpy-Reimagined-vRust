"""HRRR Zarr point-backend regressions."""

from __future__ import annotations

from datetime import datetime, timezone
import json

import numpy as np
from numcodecs import get_codec
import pytest

from sharpmod.hrrr_zarr import (
    decode_zarr_point,
    discover_pressure_plan,
    fetch_hrrr_zarr_point,
    hrrr_grid_index,
)


def _array_metadata(shape=(1059, 1799), chunks=(150, 150)):
    return {
        "shape": list(shape),
        "chunks": list(chunks),
        "dtype": "<f4",
        "compressor": {
            "id": "blosc", "cname": "lz4", "clevel": 5,
            "shuffle": 1, "blocksize": 0,
        },
        "fill_value": -9999.0,
        "filters": None,
        "order": "C",
        "zarr_format": 2,
    }


def test_hrrr_projection_maps_known_conus_point():
    iy, ix, selected_lat, selected_lon = hrrr_grid_index(35.0, -97.0)

    assert (iy, ix) == (399, 914)
    assert selected_lat == pytest.approx(35.009, abs=0.03)
    assert selected_lon == pytest.approx(-97.013, abs=0.03)


def test_metadata_plan_keeps_all_levels_and_prunes_equivalent_fields():
    metadata = {}
    for level in ("1000mb", "975mb"):
        for field in (
            "HGT", "TMP", "RH", "SPFH", "UGRD", "VGRD",
            "VVEL", "ABSV",
        ):
            key = f"{level}/{field}/{level}/{field}/.zarray"
            metadata[key] = _array_metadata()

    plan = discover_pressure_plan(metadata)

    assert plan.levels == (1000.0, 975.0)
    assert plan.fields == (
        "HGT", "TMP", "UGRD", "VGRD", "RH", "VVEL", "ABSV"
    )
    assert len(plan.arrays) == len(plan.levels) * len(plan.fields)


def test_decode_zarr_point_handles_compressed_chunk():
    metadata = _array_metadata(shape=(2, 2), chunks=(2, 2))
    values = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype="<f4")
    payload = get_codec(metadata["compressor"]).encode(values.tobytes())

    selected = decode_zarr_point(payload, metadata, iy=1, ix=0)

    assert selected == 3.0


def test_sparse_missing_chunk_is_treated_as_fill_value(monkeypatch, tmp_path):
    metadata = {}
    for field in ("HGT", "TMP", "RH", "UGRD", "VGRD"):
        key = f"1000mb/{field}/1000mb/{field}/.zarray"
        metadata[key] = _array_metadata(shape=(2, 2), chunks=(2, 2))
    consolidated = json.dumps({"metadata": metadata}).encode("utf-8")
    values = np.asarray([[280.0, 281.0], [282.0, 283.0]], dtype="<f4")
    payload = get_codec(metadata[
        "1000mb/TMP/1000mb/TMP/.zarray"]["compressor"]
    ).encode(values.tobytes())

    def get_bytes(url):
        if url.endswith("/.zmetadata"):
            return consolidated
        if "/HGT/" in url:
            # Sparse Zarr stores omit chunks that contain only fill values.
            raise FileNotFoundError(url)
        return payload

    monkeypatch.setattr(
        "sharpmod.hrrr_zarr.hrrr_grid_index",
        lambda _lat, _lon: (0, 0, 35.0, -97.0),
    )
    dataset, _source = fetch_hrrr_zarr_point(
        datetime(2026, 7, 15, 7, tzinfo=timezone.utc),
        0, 35.0, -97.0, cache_dir=tmp_path, get_bytes=get_bytes,
    )

    assert np.isnan(float(dataset["gh"].values[0, 0, 0]))
    assert float(dataset["t"].values[0, 0, 0]) == pytest.approx(280.0)
