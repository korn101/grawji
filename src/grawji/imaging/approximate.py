"""Approximate recipes without a camera, from a body's calibration tables.

A calibration bundle holds tables measured on one body by rendering
synthetic charts through it. To approximate a recipe on a photo, its
built-in JPEG is taken back to linear scene values by undoing the recipe
it was shot with, then taken forward through the target recipe:

- undo: the shot's setting tables in reverse, the film simulation's
  inverse table at the shot's dynamic range, the shot's WB shift;
- redo: the target's white balance, WB shift and exposure as gains, the
  B&W tone curves, the film simulation's forward table at the target's
  dynamic range (capped at the shot's), then its setting tables.

The whole chain is a per-colour function, so it is folded into one 3D
lookup table that Pillow applies to the image in a single pass. Grain,
clarity, sharpness and noise reduction are not reproduced, and a B&W shot
cannot be taken back to colour.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageFilter

from grawji.recipe import Recipe

Array = NDArray[np.float32]

_DR = {"DR100": 100, "DR200": 200, "DR400": 400}
_TABLE_SIZE = 33
# uint16 storage: value = (stored - offset) / scale
_STORE_OFFSET = 10000
_STORE_SCALE = 40000
_LUMA = np.array([0.3, 0.6, 0.1])
_LUMA_FLOOR = 2.0**-16
_AUTO_WB = ("Auto", "AsShot")


def data_dir() -> Path:
    """Return grawji's data directory (honors XDG_DATA_HOME)."""
    base = os.environ.get("XDG_DATA_HOME") or str(
        Path.home() / ".local" / "share"
    )
    return Path(base) / "grawji"


def bundle_path(model: str) -> Path:
    """Where the calibration bundle of a body lives."""
    return data_dir() / "filmsim" / f"{model}.npz"


@dataclass(frozen=True)
class Shot:
    """What the camera applied when the photo was taken.

    Attributes:
        recipe: The recipe recorded in the photo's metadata.
        levels: The absolute R/G, 1, B/G white balance levels applied,
            or None when unknown (fixed-WB targets then keep the shot's).
    """

    recipe: Recipe
    levels: Array | None


def lut3(img: Array, lut: Array) -> Array:
    """Trilinear lookup of float RGB values in [0, 1]."""
    n = lut.shape[0]
    flat = img.reshape(-1, 3)
    p = np.clip(flat, 0, 1) * (n - 1)
    i0 = np.minimum(p.astype(np.int32), n - 2)
    f = (p - i0).astype(np.float32)
    out = np.zeros(flat.shape, np.float32)
    r, g, b = i0[:, 0], i0[:, 1], i0[:, 2]
    for dr in (0, 1):
        wr = f[:, 0:1] if dr else 1 - f[:, 0:1]
        for dg in (0, 1):
            wrg = wr * (f[:, 1:2] if dg else 1 - f[:, 1:2])
            for db in (0, 1):
                w = wrg * (f[:, 2:3] if db else 1 - f[:, 2:3])
                out += w * lut[r + dr, g + dg, b + db]
    return out.reshape(img.shape)


def _gain(table: dict[str, float], step: int) -> float:
    """A WB shift step's gain, log-interpolated between measured steps."""
    items = sorted((int(k), v) for k, v in table.items())
    xs = np.array([k for k, _ in items], dtype=float)
    ys = np.log([v for _, v in items])
    return float(np.exp(np.interp(step, xs, ys)))


def setting_keys(recipe: Recipe) -> list[str]:
    """The measured setting tables a recipe uses, in application order."""
    keys = []
    if recipe.highlights:
        keys.append(f"H{recipe.highlights:+g}")
    if recipe.shadows:
        keys.append(f"S{recipe.shadows:+g}")
    if recipe.color:
        keys.append(f"col{recipe.color:+d}")
    if (recipe.color_chrome, recipe.color_chrome_blue) != ("Off", "Off"):
        keys.append(f"cc{recipe.color_chrome}{recipe.color_chrome_blue}")
    return keys


class Calibration:
    """One body's tables, loaded from its bundle."""

    def __init__(self, path: Path) -> None:
        """Load the bundle at path."""
        with np.load(path) as z:
            self.meta: dict[str, Any] = json.loads(bytes(z["meta"]).decode())
            self._t: dict[str, Array] = {}
            for key in z.files:
                if key == "meta":
                    continue
                value = z[key]
                self._t[key] = (
                    value.astype(np.float32)
                    if key.startswith("tone/")
                    else (value.astype(np.float32) - _STORE_OFFSET)
                    / _STORE_SCALE
                )
        enc = self.meta["encoding"]
        self._eps = float(enc["eps"])
        self._lo = float(np.log2(self._eps))
        self._hi = float(np.log2(enc["s_max"] + self._eps))
        self.colour_sims = frozenset(self.meta["colour_sims"])
        self.sims = frozenset(self.meta["sims"])

    def _encode(self, s: Array) -> Array:
        u = (np.log2(np.maximum(s, 0) + self._eps) - self._lo) / (
            self._hi - self._lo
        )
        return np.clip(u, 0, 1).astype(np.float32)

    def _decode(self, u: Array) -> Array:
        s = 2.0 ** (u * (self._hi - self._lo) + self._lo) - self._eps
        return np.maximum(s, 0).astype(np.float32)

    def shift_gain(self, r: int, b: int) -> Array:
        """Scene gains of a WB shift."""
        return np.array(
            [
                _gain(self.meta["shift_r"], r),
                1.0,
                _gain(self.meta["shift_b"], b),
            ],
            dtype=np.float32,
        )

    def preset_levels(self, mode: str, kelvin: int = 5500) -> Array | None:
        """Absolute levels of a fixed WB mode, or None if not measured."""
        base = np.array(self.meta["carrier_levels"], dtype=np.float32)
        if mode == "Temperature":
            ks = sorted(int(k) for k in self.meta["kelvin"])
            g = np.array([self.meta["kelvin"][str(k)] for k in ks])
            lk, lks = np.log(kelvin), np.log(ks)
            return base * np.exp(
                [
                    np.interp(lk, lks, np.log(g[:, 0])),
                    0.0,
                    np.interp(lk, lks, np.log(g[:, 2])),
                ]
            ).astype(np.float32)
        preset = self.meta["presets"].get(mode)
        return None if preset is None else base * np.array(preset)

    def can_approximate(self, shot: Recipe, target: Recipe) -> bool:
        """Whether both recipes are covered: a colour shot, a known sim."""
        return (
            shot.film_simulation in self.colour_sims
            and target.film_simulation in self.sims
        )

    def _undo(self, img: Array, shot: Shot) -> Array:
        """Built-in JPEG values -> scene, in the shot's own WB levels."""
        r = shot.recipe
        sim = r.film_simulation
        x = img
        for key in reversed(setting_keys(r)):
            lut = self._t.get(f"unset/{sim}/{key}")
            if lut is not None:
                x = lut3(x, lut)
        dr = r.dynamic_range if r.dynamic_range in _DR else "DR100"
        s = self._decode(lut3(x, self._t[f"inv/{sim}/{dr}"]))
        return s / self.shift_gain(r.wb_shift_r, r.wb_shift_b)

    def _tone(self, sim: str, keys: list[str], s: Array) -> Array:
        """B&W sims: highlight/shadow tone as scene exposure curves."""
        y = np.log2(np.maximum(s @ _LUMA, _LUMA_FLOOR))
        la = np.zeros_like(y)
        for key in keys:
            curve = self._t.get(f"tone/{sim}/{key}")
            if curve is not None:
                la += np.interp(y, curve[0], curve[1])
        return (s * (2.0**la)[..., None]).astype(np.float32)

    def _redo(self, s: Array, shot: Shot, target: Recipe) -> Array:
        """Scene -> the target recipe's look."""
        sim = target.film_simulation
        shot_dr = shot.recipe.dynamic_range
        dr = target.dynamic_range
        if _DR.get(dr, 0) > _DR.get(shot_dr, 100) or dr not in _DR:
            dr = shot_dr if shot_dr in _DR else "DR100"
        scale = np.ones(3, dtype=np.float32)
        if target.white_balance not in _AUTO_WB and shot.levels is not None:
            levels = self.preset_levels(
                target.white_balance, target.color_temp
            )
            if levels is not None:
                scale = levels / shot.levels
        s = s * scale * self.shift_gain(target.wb_shift_r, target.wb_shift_b)
        # The built-in JPEG already carries the shot's own EV.
        s = s * np.float32(2.0 ** (target.exposure - shot.recipe.exposure))
        keys = setting_keys(target)
        if sim not in self.colour_sims:
            s = self._tone(sim, keys, s)
        x = lut3(self._encode(s), self._t[f"fwd/{sim}/{dr}"])
        if sim in self.colour_sims:
            for key in keys:
                lut = self._t.get(f"set/{sim}/{key}")
                if lut is not None:
                    x = lut3(x, lut)
        return np.clip(x, 0, 1)

    def look_table(self, shot: Shot, target: Recipe) -> ImageFilter.Color3DLUT:
        """The whole shot -> target conversion as one Pillow 3D LUT."""
        g = np.linspace(0, 1, _TABLE_SIZE, dtype=np.float32)
        grid = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1)
        out = self._redo(self._undo(grid, shot), shot, target)
        # Pillow wants blue as the slowest axis and red as the fastest.
        table = out.transpose(2, 1, 0, 3).reshape(-1)
        return ImageFilter.Color3DLUT(_TABLE_SIZE, table.tolist())

    def shot(
        self, recipe: Recipe, auto_levels: tuple[int, int, int] | None
    ) -> Shot:
        """The shot as the camera applied it.

        Auto shots use the levels the camera recorded; a fixed mode its
        measured preset levels.
        """
        levels: Array | None = None
        if recipe.white_balance in _AUTO_WB:
            if auto_levels is not None:
                g, r, b = auto_levels
                levels = np.array([r / g, 1.0, b / g], dtype=np.float32)
        else:
            levels = self.preset_levels(
                recipe.white_balance, recipe.color_temp
            )
        return Shot(recipe, levels)


def apply_table(
    rgb: NDArray[np.uint8], table: ImageFilter.Color3DLUT
) -> NDArray[np.uint8]:
    """An RGB image through a look table."""
    out = Image.fromarray(rgb, "RGB").filter(table)
    return np.asarray(out, dtype=np.uint8)


@lru_cache(maxsize=4)
def _load(path: str, mtime: float) -> Calibration:
    del mtime  # part of the cache key, so a new bundle is picked up
    return Calibration(Path(path))


def calibration_for(model: str | None) -> Calibration | None:
    """The calibration bundle of a body, if one is installed."""
    if not model:
        return None
    path = bundle_path(model)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    try:
        return _load(str(path), mtime)
    except (OSError, ValueError, KeyError):
        return None


__all__ = [
    "Calibration",
    "Shot",
    "apply_table",
    "bundle_path",
    "calibration_for",
    "data_dir",
    "setting_keys",
]
