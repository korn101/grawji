"""Develop a RAF's raw data the way the camera's engine starts out.

A body's texture calibration (its ``<model>.texture.json``) fixes each
step, measured by rendering synthetic RAFs through the camera:

- white balance: every site times its colour's level (R/G, 1, B/G);
- demosaic: X-Trans sites to RGB, with luminance from every site and
  colour from smooth local means, so fine neutral detail stays neutral;
- geometry: the engine's distortion correction and crop to its output
  size, as a position model from raw to output pixels;
- lens gain: the engine brightens the frame radially;
- scene: levels as fractions of the green clip, times the DR push.

Previews are developed straight at a fraction of the output size: each
f x f block of sites gives one pixel's luminance, and the colour means
are evaluated only at block centres. That matches developing at full
size and averaging f x f blocks to within a small fraction of a ΔE, at
about a fifth of the cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rawpy
from numpy.typing import NDArray
from rawpy._rawpy import LibRawError

Array = NDArray[np.float32]

WHITE = 16383
PUSH = {"DR100": 1.0, "DR200": 2.0, "DR400": 4.0}
# Keys cubic for resampling blocks: -0.75 keeps the detail a full-size
# development has once averaged down.
_CUBIC = -0.75
_POSITION_STEPS = 4


@dataclass(frozen=True)
class Mosaic:
    """A RAF's raw frame.

    Attributes:
        data: Raw values, rows x columns.
        pattern: The 6x6 X-Trans tile at data[0, 0] (0 R, 1 G, 2 B).
        black: Black level.
    """

    data: NDArray[np.uint16]
    pattern: NDArray[np.uint8]
    black: float

    def colors(self) -> NDArray[np.uint8]:
        """The colour of every site."""
        h, w = self.data.shape
        reps = (-(-h // 6), -(-w // 6))
        return np.tile(self.pattern, reps)[:h, :w]


@dataclass(frozen=True)
class Geometry:
    """Where the engine's output pixels come from.

    Attributes:
        visible: The raw area the engine uses: top, left, height, width.
        out: Output size (h, w).
        gain: Lens gain polynomial in r^2, r^4, r^6 (r by half-diagonal).
        position: Rows mapping undistorted to output offsets from the
            centre, for y and x: [1, ey, ex, ey r^2, ex r^2] coefficients.
        offset: Sub-pixel offset of raw sample positions.
        chroma_sigma: Width of the demosaic's colour means, in pixels.
    """

    visible: tuple[int, int, int, int]
    out: tuple[int, int]
    gain: tuple[float, float, float]
    position: NDArray[np.float64]
    offset: float
    chroma_sigma: float

    @classmethod
    def from_meta(cls, meta: dict[str, Any]) -> Geometry:
        """From the develop section of a texture file."""
        top, left, height, width = (int(v) for v in meta["visible"])
        a, b, c = (float(v) for v in meta["gain"])
        return cls(
            visible=(top, left, height, width),
            out=(int(meta["out"][0]), int(meta["out"][1])),
            gain=(a, b, c),
            position=np.array(meta["position"], dtype=np.float64),
            offset=float(meta["offset"]),
            chroma_sigma=float(meta["chroma_sigma"]),
        )

    def fits(self, rows: int, columns: int) -> bool:
        """Whether a raw frame of this size holds the visible area."""
        top, left, height, width = self.visible
        return top + height <= rows and left + width <= columns


@dataclass(frozen=True)
class RawInfo:
    """What a RAF says about its raw data, read without decoding it.

    Attributes:
        rows: Raw frame height.
        columns: Raw frame width.
        as_shot: The white balance levels (G, R, B) the camera applied,
            its WB shift included.
    """

    rows: int
    columns: int
    as_shot: tuple[float, float, float]


def raw_info(path: Path | str) -> RawInfo:
    """A RAF's raw frame size and white balance.

    Raises:
        ValueError: The file has no raw data libraw can read.
    """
    try:
        with rawpy.imread(str(path)) as raw:
            sizes = raw.sizes
            # libraw reports the stored triple as (R, G, B, G2).
            r, g, b = (float(v) for v in raw.camera_whitebalance[:3])
            return RawInfo(
                int(sizes.raw_height), int(sizes.raw_width), (g, r, b)
            )
    except LibRawError as exc:
        raise ValueError(f"cannot read the raw data: {exc}") from exc


def read_raf(path: Path | str) -> Mosaic:
    """A RAF's raw frame.

    Raises:
        ValueError: The file has no raw data libraw can decode.
    """
    try:
        with rawpy.imread(str(path)) as raw:
            data = np.asarray(raw.raw_image, dtype=np.uint16).copy()
            desc = raw.color_desc.decode()
            index = np.array(["RGB".index(c) for c in desc])
            colors = index[raw.raw_colors[:6, :6]].astype(np.uint8)
            black = float(raw.black_level_per_channel[0])
    except LibRawError as exc:
        raise ValueError(f"cannot decode the raw data: {exc}") from exc
    return Mosaic(data, colors, black)


# ---- filters (numpy only) ---------------------------------------------


def gaussian_kernel(sigma: float) -> Array:
    """A normalised 1D Gaussian reaching 3 sigma."""
    r = max(1, int(np.ceil(3 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-(x**2) / (2 * sigma**2))
    return (k / k.sum()).astype(np.float32)


def convolve_axis(
    img: NDArray[Any], kernel: Array, axis: int, step: int = 1, start: int = 0
) -> Array:
    """1D convolution along an axis, edges mirrored.

    With a step, only the outputs at start, start + step, ... are made.
    """
    r = len(kernel) // 2
    pad = [(0, 0)] * img.ndim
    pad[axis] = (r, r)
    p = np.pad(img, pad, mode="reflect")
    n = (img.shape[axis] - start + step - 1) // step
    shape = list(img.shape)
    shape[axis] = n
    out = np.zeros(shape, np.float32)
    for i, k in enumerate(kernel):
        taps = [slice(None)] * img.ndim
        taps[axis] = slice(start + i, start + i + step * (n - 1) + 1, step)
        out += k * p[tuple(taps)]
    return out


def blur(img: NDArray[Any], sigma: float) -> Array:
    """Separable Gaussian blur of the first two axes."""
    k = gaussian_kernel(sigma)
    return convolve_axis(convolve_axis(img, k, 0), k, 1)


def _keys(t: Array) -> tuple[NDArray[Any], ...]:
    """Keys cubic weights of the taps at -1, 0, 1, 2 for fraction t."""
    a = _CUBIC
    t2, t3 = t * t, t * t * t
    return (
        a * (t3 - 2 * t2 + t),
        (a + 2) * t3 - (a + 3) * t2 + 1,
        -(a + 2) * t3 + (2 * a + 3) * t2 - a * t,
        -a * (t3 - t2),
    )


def sample(img: Array, y: NDArray[Any], x: NDArray[Any]) -> Array:
    """Cubic samples of a 2D image at float positions, edges clamped."""
    h, w = img.shape[:2]
    y = np.clip(y, 0, h - 1)
    x = np.clip(x, 0, w - 1)
    y0 = np.floor(y).astype(np.int32)
    x0 = np.floor(x).astype(np.int32)
    wy = _keys((y - y0).astype(np.float32))
    wx = _keys((x - x0).astype(np.float32))
    cols = [np.clip(x0 + j - 1, 0, w - 1) for j in range(4)]
    out = np.zeros(y.shape, np.float32)
    for i in range(4):
        rows = np.clip(y0 + i - 1, 0, h - 1)
        line = np.zeros(y.shape, np.float32)
        for j in range(4):
            line += wx[j] * img[rows, cols[j]]
        out += wy[i] * line
    return out


# ---- development --------------------------------------------------------


def raw_positions(
    geometry: Geometry, yy: NDArray[np.float64], xx: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Where output pixel positions (yy, xx) lie on the visible raw grid."""
    h, w = geometry.out
    vh, vw = geometry.visible[2], geometry.visible[3]
    ty, tx = yy - h / 2, xx - w / 2
    ey, ex = ty.copy(), tx.copy()
    norm = (h / 2) ** 2 + (w / 2) ** 2
    # The model maps undistorted to output positions; invert it by
    # fixed-point steps (it is close to the identity).
    for _ in range(_POSITION_STEPS):
        r2 = (ey**2 + ex**2) / norm
        terms = np.stack([np.ones_like(ey), ey, ex, ey * r2, ex * r2], -1)
        ey -= terms @ geometry.position[0] - ty
        ex -= terms @ geometry.position[1] - tx
    return (
        (ey + h / 2) * vh / h + geometry.offset,
        (ex + w / 2) * vw / w + geometry.offset,
    )


def _gain(
    geometry: Geometry, fy: NDArray[np.float64], fx: NDArray[np.float64]
) -> Array:
    """The lens gain at offsets (fy, fx) from the centre.

    The offsets are fractions of the diagonal, so the half-diagonal is 0.5.
    """
    r2 = (fy**2 + fx**2) / 0.25
    a, b, c = geometry.gain
    return (1 + a * r2 + b * r2**2 + c * r2**3).astype(np.float32)


def gain_map(geometry: Geometry, height: int, width: int) -> Array:
    """The engine's radial lens gain over a frame of the given size."""
    norm = np.hypot(height, width)
    fy = (np.arange(height) + 0.5 - height / 2)[:, None] / norm
    fx = (np.arange(width) + 0.5 - width / 2)[None, :] / norm
    return _gain(geometry, fy, fx)


def develop(
    mosaic: Mosaic,
    levels: NDArray[Any],
    geometry: Geometry,
    dynamic_range: str = "DR100",
    factor: int = 3,
) -> Array:
    """Scene levels of the output frame at 1/factor size.

    Relative to the DR100 clip. The result is what developing at full size
    and averaging factor x factor blocks gives.

    Args:
        mosaic: The raw data.
        levels: White balance levels (R/G, 1, B/G) the shot was taken with,
            without its WB shift.
        geometry: The body's output geometry.
        dynamic_range: The shot's DR setting, which sets its push.
        factor: How many output pixels a side one pixel stands for.
    """
    f = factor
    top, left, vh, vw = geometry.visible
    colors = mosaic.colors()[top : top + vh, left : left + vw]
    lv = np.asarray(levels, np.float32)
    site = (
        mosaic.data[top : top + vh, left : left + vw].astype(np.float32)
        - mosaic.black
    ) * lv[colors]
    bh, bw = vh // f, vw // f

    def blocks(img: NDArray[Any]) -> Array:
        cut = img[: bh * f, : bw * f].reshape(bh, f, bw, f)
        return cut.sum(axis=(1, 3), dtype=np.float32)

    # Colour: the demosaic's Gaussian colour means at block centres, wide
    # enough to stand for their average over the block.
    k = gaussian_kernel(np.sqrt(geometry.chroma_sigma**2 + (f * f - 1) / 12))
    c0 = (f - 1) // 2

    def at_centres(img: Array) -> Array:
        rows = convolve_axis(img, k, 1, f, c0)
        return convolve_axis(rows, k, 0, f, c0)[:bh, :bw]

    means, counts = [], []
    for c in range(3):
        mask = (colors == c).astype(np.float32)
        num = at_centres(site * mask)
        means.append(num / np.maximum(at_centres(mask), 1e-6))
        counts.append(blocks(mask))
    dr, db = means[0] - means[1], means[2] - means[1]
    # Luminance: every site of the block, less the colour differences its
    # red and blue sites carry.
    g = (blocks(site) - counts[0] * dr - counts[2] * db) / (f * f)
    rgb = np.stack([g + dr, g, g + db], -1)
    h, w = geometry.out
    ph, pw = h // f, w // f
    yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float64) * f + (f - 1) / 2
    ry, rx = raw_positions(geometry, yy, xx)
    by, bx = (ry - (f - 1) / 2) / f, (rx - (f - 1) / 2) / f
    out = np.stack([sample(rgb[..., c], by, bx) for c in range(3)], -1)
    # The gain at block centres (within 1e-6 of its block average).
    norm = np.hypot(h, w)
    gain = _gain(
        geometry, (yy + 0.5 - h / 2) / norm, (xx + 0.5 - w / 2) / norm
    )
    full = WHITE - mosaic.black
    push = PUSH.get(dynamic_range, 1.0)
    return (out * (gain[..., None] * push / full)).astype(np.float32)


__all__ = [
    "PUSH",
    "Geometry",
    "Mosaic",
    "RawInfo",
    "blur",
    "convolve_axis",
    "develop",
    "gain_map",
    "gaussian_kernel",
    "raw_info",
    "raw_positions",
    "read_raf",
    "sample",
]
