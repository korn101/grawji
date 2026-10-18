"""Noise reduction, clarity, sharpening and grain, on synthetic images."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from grawji.imaging.texture import (
    clarity,
    denoise,
    grain,
    luma,
    raw_noise,
    sharpen,
)
from tests.approx_support import texture

TEXTURE: dict[str, Any] = texture()


def test_settings_at_their_defaults_change_nothing() -> None:
    """Clarity 0, sharpness -4 and grain Off leave the image alone."""
    img = np.random.default_rng(0).uniform(0, 255, (40, 60, 3))
    img = img.astype(np.float32)
    rng = np.random.default_rng(1)
    assert np.array_equal(clarity(img, 0, TEXTURE["clarity"]), img)
    assert np.array_equal(sharpen(img, -4, TEXTURE["sharpness"]), img)
    assert np.array_equal(
        grain(img, "Off", "Small", "Provia", TEXTURE["grain"], rng=rng), img
    )


def _edge() -> np.ndarray:
    img = np.full((40, 40, 3), 90.0, np.float32)
    img[:, 20:] = 150.0
    return img


def test_sharpening_grows_with_the_level() -> None:
    """More sharpness, more overshoot at an edge; none on flat areas."""
    overshoot = [
        sharpen(_edge(), s, TEXTURE["sharpness"])[20, 20:24, 1].max() - 150
        for s in (-3, 0, 4)
    ]
    assert 0 < overshoot[0] < overshoot[1] < overshoot[2]
    flat = np.full((20, 20, 3), 120.0, np.float32)
    assert np.allclose(sharpen(flat, 4, TEXTURE["sharpness"]), flat)


def test_clarity_works_on_wide_detail_both_ways() -> None:
    """Positive clarity adds local contrast, negative takes it away."""
    x = np.arange(240)
    row = 100 + 30 * np.sin(2 * np.pi * x / 60)
    img = np.repeat(row[None, :, None], 8, 0).repeat(3, 2).astype(np.float32)
    base = luma(img)[4, 40:200].std()
    up = luma(clarity(img, 5, TEXTURE["clarity"]))[4, 40:200].std()
    down = luma(clarity(img, -5, TEXTURE["clarity"]))[4, 40:200].std()
    assert down < base * 0.95 < base * 1.05 < up


def test_grain_follows_its_strength_curve() -> None:
    """Luma-only noise of the curve's strength; Acros's is stronger."""
    img = np.full((300, 300, 3), 128.0, np.float32)
    params = TEXTURE["grain"]
    rng = np.random.default_rng(2)
    added = grain(img, "Strong", "Small", "Provia", params, rng=rng) - img
    assert np.allclose(added[..., 0], added[..., 2])
    assert abs(added.mean()) < 0.2
    assert added[..., 0].std() == pytest.approx(4.0, rel=0.03)
    acros = grain(img, "Strong", "Small", "Acros", params, rng=rng) - img
    assert acros.std() > added.std()
    half = grain(img, "Strong", "Small", "Provia", params, rng=rng, scale=0.1)
    assert (half - img).std() == pytest.approx(0.4, rel=0.05)


def test_large_grain_is_clumped() -> None:
    """Neighbouring pixels share Large grain, not Small."""
    img = np.full((200, 200, 3), 128.0, np.float32)
    rng = np.random.default_rng(3)
    for size, lo, hi in (("Small", -0.1, 0.1), ("Large", 0.6, 0.9)):
        n = grain(img, "Strong", size, "Provia", TEXTURE["grain"], rng=rng)
        n = (n - img)[..., 0]
        n -= n.mean()
        corr = (n[:, 1:] * n[:, :-1]).mean() / n.var()
        assert lo < corr < hi, size


def _noisy_scene(iso: float = 3200, h: int = 96, w: int = 96) -> np.ndarray:
    """A grey scene with the sensor's noise at an ISO, and an edge."""
    p = TEXTURE["noise"]
    a, b = raw_noise(p, iso)
    full = p["full"]
    s = np.full((h, w, 3), 0.08, np.float32)
    s[:, w // 2 :] *= 3
    sd = np.sqrt(a * s * full + b) / full
    rng = np.random.default_rng(0)
    return (s + rng.standard_normal(s.shape) * sd).astype(np.float32)


def _denoise(scene: np.ndarray, level: int) -> np.ndarray:
    return denoise(
        scene,
        level,
        TEXTURE["noise"],
        iso=3200,
        push=1,
        levels=np.ones(3),
        gain=np.ones(scene.shape[:2], np.float32),
    )


def test_noise_reduction_grows_with_the_level() -> None:
    """Higher NR leaves less noise on a flat area."""
    s = _noisy_scene()
    flat = (slice(10, 86), slice(8, 40))
    noise = [_denoise(s, level)[flat][..., 1].std() for level in (4, 0, -4)]
    assert noise[0] < noise[1] < noise[2] < s[flat][..., 1].std()


def test_noise_reduction_keeps_edges() -> None:
    """The step between the two halves survives the strongest NR."""
    out = _denoise(_noisy_scene(), 4)
    step = out[:, 52:60, 1].mean() - out[:, 36:44, 1].mean()
    assert step == pytest.approx(0.16, rel=0.05)
