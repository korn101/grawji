"""Approximations developed from a RAF's raw data, without a camera."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

gi = pytest.importorskip("gi")
gi.require_version("GdkPixbuf", "2.0")

from gi.repository import GdkPixbuf, GLib

from grawji.controllers import approximation
from grawji.controllers.approximation import (
    Approximation,
    Prepared,
    RawSource,
    prepare,
)
from grawji.imaging.approximate import Calibration
from grawji.imaging.develop import Mosaic, RawInfo
from grawji.imaging.pixbufs import rgb_array
from grawji.recipe import Recipe
from tests.approx_support import (
    RAW_H,
    RAW_W,
    mosaic,
    write_bundle,
    write_texture,
)

RAF = "a.RAF"


@pytest.fixture
def cal(tmp_path: Path) -> Calibration:
    """The test bundle with its texture calibration."""
    bundle = write_bundle(tmp_path / "X100V.npz")
    write_texture(bundle)
    return Calibration(bundle)


@pytest.fixture
def reads(monkeypatch: pytest.MonkeyPatch, cal: Calibration) -> list[str]:
    """A calibrated body's RAF with raw data; lists each decode."""
    decoded: list[str] = []

    def read(path: str) -> Mosaic:
        decoded.append(path)
        return mosaic((300, 500, 250))

    meta = approximation.imagemeta
    monkeypatch.setattr(meta, "camera_model", lambda _p: "X100V")
    monkeypatch.setattr(meta, "iso_speed", lambda _p: 800)
    monkeypatch.setattr(meta, "exif_orientation", lambda _p: 1)
    monkeypatch.setattr(meta, "fuji_auto_wb_levels", lambda _p: None)
    monkeypatch.setattr(approximation, "calibration_for", lambda _m: cal)
    monkeypatch.setattr(
        approximation,
        "raw_info",
        lambda _p: RawInfo(RAW_H, RAW_W, (100.0, 180.0, 180.0)),
    )
    monkeypatch.setattr(approximation, "read_raf", read)
    return decoded


def _no_raw(_path: str) -> RawInfo:
    raise ValueError("no raw data")


def test_a_raf_with_raw_data_is_developed(reads: list[str]) -> None:
    """Its ISO and orientation come along, and the levels it was shot at."""
    prepared = prepare(RAF, Recipe())
    assert prepared is not None
    assert prepared.raw == RawSource(RAF, 800.0, 1)
    assert prepared.shot.levels is not None
    np.testing.assert_allclose(prepared.shot.levels, [1.8, 1.0, 1.8])
    assert not reads  # decoding waits for the first render


def test_a_bw_shot_qualifies_with_its_raw_data(
    reads: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raw data has the colour a B&W JPEG lost."""
    acros = Recipe(film_simulation="Acros")
    prepared = prepare(RAF, acros)
    assert prepared is not None
    assert prepared.raw is not None
    monkeypatch.setattr(approximation, "raw_info", _no_raw)
    assert prepare(RAF, acros) is None


def test_unreadable_raw_data_falls_back_to_the_jpeg(
    reads: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A RAF libraw cannot read, or of another size, uses its JPEG."""
    monkeypatch.setattr(approximation, "raw_info", _no_raw)
    prepared = prepare(RAF, Recipe())
    assert prepared is not None
    assert prepared.raw is None
    monkeypatch.setattr(
        approximation,
        "raw_info",
        lambda _p: RawInfo(RAW_H - 6, RAW_W, (100.0, 180.0, 180.0)),
    )
    prepared = prepare(RAF, Recipe())
    assert prepared is not None
    assert prepared.raw is None


def _approximation(prepared: Prepared) -> tuple[Approximation, list[Any]]:
    results: list[Any] = []
    approx = Approximation(
        on_done=lambda _g, _s, pixbuf: results.append(pixbuf),
        on_error=lambda _g, _s, error: results.append(error),
    )
    jpeg = GdkPixbuf.Pixbuf.new(GdkPixbuf.Colorspace.RGB, False, 8, 6, 4)
    jpeg.fill(0x808080FF)
    assert approx.load(prepared, jpeg)
    return approx, results


def _render(approx: Approximation, results: list[Any], recipe: Recipe) -> Any:
    count = len(results)
    approx.render(0, recipe)
    context = GLib.MainContext.default()
    deadline = time.monotonic() + 10
    while len(results) == count:
        assert time.monotonic() < deadline, "the render never finished"
        context.iteration(may_block=False)
        time.sleep(0.005)
    result = results[-1]
    assert not isinstance(result, str), result
    return rgb_array(result)


def test_recipes_render_on_the_developed_raw_data(reads: list[str]) -> None:
    """The raw data is decoded once; each recipe's texture applies."""
    prepared = prepare(RAF, Recipe())
    assert prepared is not None
    approx, results = _approximation(prepared)
    plain = _render(approx, results, Recipe()).astype(float)
    assert plain.shape == (RAW_H // 3, RAW_W // 3, 3)
    assert plain[4:-4, 4:-4].std(axis=(0, 1)).max() < 1
    grainy = _render(approx, results, Recipe(grain="Strong")).astype(float)
    # Strong grain is 4 code values at full size, a third of it here.
    assert (grainy - plain)[4:-4, 4:-4].std() == pytest.approx(4 / 3, rel=0.15)
    again = _render(approx, results, Recipe(grain="Strong"))
    assert np.array_equal(again, grainy)  # the grain does not crawl
    brighter = _render(approx, results, Recipe(exposure=1.0)).astype(float)
    assert brighter.mean() > plain.mean() + 10
    assert reads == [RAF]


def test_the_render_is_turned_upright(
    reads: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A portrait shot shows as its EXIF orientation says."""
    monkeypatch.setattr(
        approximation.imagemeta, "exif_orientation", lambda _p: 6
    )
    prepared = prepare(RAF, Recipe())
    assert prepared is not None
    approx, results = _approximation(prepared)
    rgb = _render(approx, results, Recipe())
    assert rgb.shape == (RAW_W // 3, RAW_H // 3, 3)
