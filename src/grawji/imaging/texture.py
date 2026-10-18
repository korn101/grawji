"""Noise reduction, clarity, sharpening and grain, as the camera does them.

Noise reduction works on the developed scene, before the colour tables.
It was fitted on real photos from ISO 160 to 3200 rendered at every NR
level, and needs the sensor's noise at the shot's ISO:

- Noise reduction: luma detail in two bands (Y - blur(Y, 1), and
  blur(Y, 1) - blur(Y, 2.5)) is kept in proportion to how far its local
  power rises above the expected noise power (spectral subtraction with a
  strength per ISO and NR level); colour differences are blurred.

Clarity, sharpening and grain were measured by rendering synthetic charts
at every level. They act on luma (Rec. 601 weights) in output code values
(0..255) and add the same change to R, G and B. Sizes are in sensor
pixels: `scale` is the image's width over the camera's output width, so a
smaller image gets a proportionally smaller effect.

- Clarity: a wide local-contrast term, blur(Y, 1 px) - blur(Y, 110 px)
  plus a little of the finest detail, weighted by the pixel's own level
  (full in shadows and mid tones, none in the highlights).
- Sharpening: an unsharp mask on Y - blur(Y, 1.2 px), with separate soft
  limits for bright and dark halos and slightly more in the shadows.
  Sharpness -4 is none: the developed image already has the camera's
  detail at -4.
- Grain: random per render. Small grain is uniform white noise per pixel;
  Large grain is that noise blurred by 1 px. Its strength follows the
  output level, and Acros grain is stronger.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from grawji.imaging.develop import blur

Array = NDArray[np.float32]
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)
ACROS = ("Acros", "AcrosYe", "AcrosR", "AcrosG")


def luma(img: Array) -> Array:
    """Rec. 601 luma of an RGB image in code values."""
    return (img @ LUMA).astype(np.float32)


def _wide_blur(y: Array, sigma: float) -> Array:
    """A Gaussian blur of a wide sigma, done at a reduced size."""
    f = max(1, int(sigma // 8))
    if f == 1:
        return blur(y, sigma)
    h, w = y.shape
    ph, pw = -(-h // f) * f, -(-w // f) * f
    p = np.pad(y, ((0, ph - h), (0, pw - w)), mode="edge")
    small = p.reshape(ph // f, f, pw // f, f).mean(axis=(1, 3))
    small = blur(small.astype(np.float32), np.sqrt(sigma**2 - f**2 / 12) / f)
    return _upsample(_upsample(small, h, f, 0), w, f, 1)


def _upsample(img: Array, n: int, f: int, axis: int) -> Array:
    """Linear interpolation of an axis back to n samples (block size f)."""
    pos = np.clip((np.arange(n) + 0.5) / f - 0.5, 0, img.shape[axis] - 1)
    i0 = np.minimum(np.floor(pos).astype(np.int32), img.shape[axis] - 2)
    t = (pos - i0).astype(np.float32)
    a = np.take(img, i0, axis=axis)
    b = np.take(img, i0 + 1, axis=axis)
    t = t[:, None] if axis == 0 else t[None, :]
    return (a * (1 - t) + b * t).astype(np.float32)


def _level_table(
    levels: dict[str, Any], level: float
) -> dict[str, Any] | None:
    """A level's parameters, interpolated between measured levels."""
    known = sorted(float(k) for k in levels)
    if level in known:
        return levels[f"{level:+g}"]
    lo = max((k for k in known if k < level), default=None)
    hi = min((k for k in known if k > level), default=None)
    if lo is None or hi is None:
        return None
    t = (level - lo) / (hi - lo)
    a, b = levels[f"{lo:+g}"], levels[f"{hi:+g}"]
    return {
        k: (np.asarray(a[k]) * (1 - t) + np.asarray(b[k]) * t).tolist()
        for k in a
    }


def clarity(
    img: Array, level: int, params: dict[str, Any], scale: float = 1.0
) -> Array:
    """Clarity -5..+5 on an output image (code values)."""
    p = _level_table(params["levels"], float(level)) if level else None
    if p is None:
        return img
    y = luma(img)
    # Detail from just above pixel level to the wide blur, plus a little of
    # the finest detail, which the camera's clarity mostly leaves alone.
    fine = blur(y, max(params["fine_sigma"] * scale, 0.3))
    wide = _wide_blur(y, params["sigma"] * scale)
    detail = fine - wide + params["fine_weight"] * (y - fine)
    centres = np.asarray(params["centres"], np.float32)
    gain = np.interp(y, centres, np.asarray(p["gain"], np.float32))
    return img + (gain * detail + float(p["offset"]))[..., None]


def sharpen(
    img: Array, level: int, params: dict[str, Any], scale: float = 1.0
) -> Array:
    """Sharpness -4..+4 on an output image developed at camera -4."""
    p = params["levels"].get(f"{level:+d}")
    if p is None:
        return img
    y = luma(img)
    d = y - blur(y, max(params["sigma"] * scale, 0.3))
    lim = np.where(
        d > 0, p["Lp"] * np.tanh(d / p["Lp"]), p["Ln"] * np.tanh(d / p["Ln"])
    )
    w = np.maximum(1 + p["k1"] * (y - 128) / 128, 0)
    return img + (w * p["A"] * lim)[..., None]


def grain(  # noqa: PLR0913
    img: Array,
    strength: str,
    size: str,
    sim: str,
    params: dict[str, Any],
    *,
    rng: np.random.Generator,
    scale: float = 1.0,
) -> Array:
    """Grain (Off, Weak, Strong; Small, Large) on an output image."""
    if strength == "Off":
        return img
    # Acros has its own, stronger grain; every other simulation shares one.
    curves = params["acros"] if sim in ACROS else params["curves"]
    curve = curves[f"{strength}/{size}"]
    y = luma(img)
    amp = np.interp(y, curve["y"], curve["amp"]).astype(np.float32)
    h, w = y.shape
    noise = rng.uniform(-np.sqrt(3), np.sqrt(3), (h, w)).astype(np.float32)
    s = min(1.0, scale)
    if size == "Large":
        noise = blur(noise, max(params["large_sigma"] * s, 0.3))
        noise /= noise.std()
    # A smaller image shows the camera's grain averaged over its pixels.
    table = np.asarray(params["downscale"][size], np.float32)
    amp *= np.float32(np.interp(s, table[:, 0], table[:, 1]))
    return img + (amp * noise)[..., None]


def raw_noise(params: dict[str, Any], iso: float) -> tuple[float, float]:
    """(a, b) of the raw noise variance a * DN + b at an analog ISO."""
    raw = params["raw"]
    isos = np.log(np.asarray(raw["iso"], float))
    li = np.log(iso)
    a = float(np.exp(np.interp(li, isos, np.log(raw["a"]))))
    b = float(np.exp(np.interp(li, isos, np.log(raw["b"]))))
    # Outside the measured ISOs, shot noise scales with the gain and read
    # noise with its square.
    for edge, inside in (
        (raw["iso"][-1], iso > raw["iso"][-1]),
        (raw["iso"][0], iso < raw["iso"][0]),
    ):
        if inside:
            a *= iso / edge
            b *= (iso / edge) ** 2
    return a, b


def noise_strength(params: dict[str, Any], iso: float, level: int) -> float:
    """The NR strength at an ISO and NR level, interpolated in log ISO."""
    st = params["strength"]
    column = [float(np.interp(level, st["nr"], row)) for row in st["k"]]
    return float(np.interp(np.log(iso), np.log(st["iso"]), column))


def denoise(  # noqa: PLR0913
    scene: Array,
    level: int,
    params: dict[str, Any],
    *,
    iso: float,
    push: float,
    levels: NDArray[Any],
    gain: Array,
    scale: float = 1.0,
) -> Array:
    """Noise reduction -4..+4 on developed scene levels.

    Args:
        scene: Scene levels (develop.develop).
        level: The NR setting.
        params: The noise section of the texture parameters.
        iso: The shot's ISO as recorded.
        push: Its DR push (1, 2 or 4); the sensor ran at iso / push.
        levels: The white balance levels the scene was developed with.
        gain: The lens gain map of the scene's frame (develop.gain_map).
        scale: The scene's width over the camera's output width.
    """
    k = noise_strength(params, iso, level)
    a, b = raw_noise(params, iso / push)
    full = params["full"]
    unit = (np.asarray(levels, np.float32) * push / full)[None, None, :]
    unit = unit * gain[..., None]
    var = unit * a * np.maximum(scene, 0) + unit**2 * b
    # A downscaled scene has averaged scale^-2 pixels of noise each.
    var_y = (var[..., 0] + 4 * var[..., 1] + var[..., 2]) / 16 * scale**2
    y = (scene[..., 0] + 2 * scene[..., 1] + scene[..., 2]) / 4
    s1, s2 = (max(v * scale, 0.3) for v in params["bands"])
    g1 = blur(y, s1)
    g2 = blur(g1, s2)
    out = g2
    sp = max(params["power_sigma"] * scale, 0.3)
    for d, f in (
        (y - g1, params["fractions"][0]),
        (g1 - g2, params["fractions"][1]),
    ):
        power = blur(d * d, sp)
        keep = np.clip(1 - k * f * var_y / np.maximum(power, 1e-12), 0, 1)
        out = out + keep * d
    chroma = blur(
        scene - y[..., None], max(params["chroma_sigma"] * scale, 0.3)
    )
    return (out[..., None] + chroma).astype(np.float32)


__all__ = [
    "clarity",
    "denoise",
    "grain",
    "luma",
    "noise_strength",
    "raw_noise",
    "sharpen",
]
