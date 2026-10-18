"""Developing raw data to preview size, on synthetic X-Trans frames."""

from __future__ import annotations

import numpy as np
import pytest

from grawji.imaging.develop import (
    Geometry,
    Mosaic,
    develop,
    gain_map,
    raw_positions,
)
from tests.approx_support import BLACK, RAW_H, RAW_W, XTRANS, mosaic, texture

FULL = 16383 - BLACK
LEVELS = np.array([5 / 3, 1.0, 2.0])
INNER = (slice(3, -3), slice(3, -3))


def _geometry(**changes: object) -> Geometry:
    meta = dict(texture()["develop"])  # type: ignore[call-overload]
    meta.update(changes)
    return Geometry.from_meta(meta)


def test_a_flat_colour_gives_white_balanced_fractions_of_clip() -> None:
    """Each colour times its level, over the clip, at a third of the size."""
    scene = develop(mosaic((300, 500, 250)), LEVELS, _geometry())
    assert scene.shape == (RAW_H // 3, RAW_W // 3, 3)
    np.testing.assert_allclose(scene[INNER], 500 / FULL, rtol=1e-4)


def test_the_dr_push_scales_the_scene() -> None:
    """DR200 and DR400 shots were exposed one and two stops down."""
    m = mosaic((300, 500, 250))
    base = develop(m, LEVELS, _geometry())
    np.testing.assert_allclose(
        develop(m, LEVELS, _geometry(), "DR400"), 4 * base, rtol=1e-5
    )


def test_each_pixel_averages_its_block() -> None:
    """A horizontal ramp keeps its level at every block's centre."""
    x = np.arange(RAW_W, dtype=float)
    ramp = np.repeat((200 + 5 * x)[None, :], RAW_H, 0)
    m = Mosaic((ramp + BLACK).astype(np.uint16), XTRANS, BLACK)
    scene = develop(m, np.ones(3), _geometry())
    centres = 200 + 5 * (np.arange(RAW_W // 3) * 3 + 1)
    np.testing.assert_allclose(
        scene[4:-4, 4:-4, 1] * FULL, np.tile(centres[4:-4], (16, 1)), rtol=1e-3
    )


def test_neutral_detail_stays_neutral() -> None:
    """Grey stripes give no colour away from the frame's edges."""
    x = np.arange(RAW_W)
    level = np.where((x // 3) % 2 == 0, 200.0, 800.0)
    stripes = np.repeat(level[None, :], RAW_H, 0)
    m = Mosaic((stripes + BLACK).astype(np.uint16), XTRANS, BLACK)
    scene = develop(m, np.ones(3), _geometry())[INNER] * FULL
    chroma = scene - scene.mean(-1, keepdims=True)
    assert np.abs(chroma).mean() < 0.05 * 300


def test_the_lens_gain_brightens_the_corners() -> None:
    """The engine's radial gain, as the geometry gives it."""
    scene = develop(
        mosaic((500, 500, 500)), np.ones(3), _geometry(gain=[0.3, 0, 0])
    )
    assert scene[0, 0, 1] > scene[12, 18, 1] * 1.2
    gain = gain_map(_geometry(gain=[0.3, 0, 0]), 24, 36)
    assert gain[11:13, 17:19] == pytest.approx(1, abs=1e-2)
    assert gain[0, 0] == pytest.approx(gain[-1, -1])


def test_the_output_grid_follows_the_position_model() -> None:
    """Without distortion an output position is its raw position."""
    yy, xx = np.mgrid[0:4, 0:6].astype(np.float64) * 3
    ry, rx = raw_positions(_geometry(), yy, xx)
    np.testing.assert_allclose(ry, yy)
    np.testing.assert_allclose(rx, xx)


def test_a_frame_must_hold_the_visible_area() -> None:
    """A smaller raw frame (another body) is not developed."""
    geometry = _geometry()
    assert geometry.fits(RAW_H, RAW_W)
    assert not geometry.fits(RAW_H - 1, RAW_W)
