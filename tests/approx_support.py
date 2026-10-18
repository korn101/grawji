"""A tiny calibration bundle whose tables are easy to reason about.

Forward and inverse tables are identities in the log-encoded scene, so
undoing a shot and redoing the same recipe gives the image back, and an
exposure step moves every value by a known amount.

The texture calibration that goes with it develops a small synthetic
X-Trans frame with no distortion, and has round-number texture settings.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from grawji.imaging.develop import Mosaic

N = 5
EPS = 2.0**-14
S_MAX = 4.0
# One stop in the encoding's 0..1 range.
STOP = 1 / (np.log2(S_MAX + EPS) - np.log2(EPS))
SIMS = ("Provia", "Velvia", "Acros")
COLOUR = ("Provia", "Velvia")


def _identity() -> np.ndarray:
    g = np.linspace(0, 1, N)
    return np.stack(np.meshgrid(g, g, g, indexing="ij"), -1)


def _store(table: np.ndarray) -> np.ndarray:
    return np.round(np.clip(table, -0.25, 1.25) * 40000 + 10000).astype(
        np.uint16
    )


def write_bundle(path: Path) -> Path:
    """Write the bundle, with a few tables that differ on purpose.

    - fwd/Provia/DR400 is a flat 0.25, to show when DR400 is used.
    - set/Provia/col+2 halves the distance to white; its unset undoes it.
    """
    ident = _identity()
    tables: dict[str, np.ndarray] = {}
    for sim in SIMS:
        for dr in ("DR100", "DR200", "DR400"):
            tables[f"fwd/{sim}/{dr}"] = _store(ident)
            if sim in COLOUR:
                tables[f"inv/{sim}/{dr}"] = _store(ident)
    tables["fwd/Provia/DR400"] = _store(np.full_like(ident, 0.25))
    tables["set/Provia/col+2"] = _store(0.5 + ident / 2)
    tables["unset/Provia/col+2"] = _store(ident * 2 - 1)
    meta = {
        "encoding": {"eps": EPS, "s_max": S_MAX},
        "carrier_levels": [1.8, 1.0, 1.8],
        "presets": {"Daylight": [1.0, 1.0, 1.0], "Shade": [2.0, 1.0, 0.5]},
        "kelvin": {"5000": [1.0, 1.0, 1.0], "10000": [2.0, 1.0, 0.5]},
        "shift_r": {"0": 1.0, "-3": 0.5, "+3": 2.0},
        "shift_b": {"0": 1.0, "-3": 0.5, "+3": 2.0},
        "sims": list(SIMS),
        "colour_sims": list(COLOUR),
        "body": "X100V",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        meta=np.frombuffer(json.dumps(meta).encode(), np.uint8),
        **tables,
    )
    return path


# The X100V's 6x6 X-Trans tile (0 R, 1 G, 2 B).
XTRANS = np.array(
    [
        [1, 1, 0, 1, 1, 2],
        [1, 1, 2, 1, 1, 0],
        [2, 0, 1, 0, 2, 1],
        [1, 1, 2, 1, 1, 0],
        [1, 1, 0, 1, 1, 2],
        [0, 2, 1, 2, 0, 1],
    ],
    dtype=np.uint8,
)
BLACK = 1000.0
# Raw frame size of the texture calibration: output pixel (y, x) is raw
# site (y, x).
RAW_H, RAW_W = 72, 108


def _levels(gain: float) -> dict[str, object]:
    return {"A": gain, "Lp": 12.0, "Ln": 12.0, "k1": 0.0}


def texture(h: int = RAW_H, w: int = RAW_W) -> dict[str, object]:
    """Texture parameters for an h x w frame."""
    curve = {"y": [0.0, 128.0, 255.0], "amp": [2.0, 2.0, 0.0]}
    strong = {"y": [0.0, 128.0, 255.0], "amp": [4.0, 4.0, 0.0]}
    acros = {"y": [0.0, 128.0, 255.0], "amp": [6.0, 6.0, 0.0]}
    return {
        "develop": {
            "visible": [0, 0, h, w],
            "out": [h, w],
            "gain": [0.0, 0.0, 0.0],
            "position": [[0, 1, 0, 0, 0], [0, 0, 1, 0, 0]],
            "offset": 0.0,
            "chroma_sigma": 2.0,
        },
        "noise": {
            "raw": {"iso": [160, 3200], "a": [0.5, 10.0], "b": [10.0, 400.0]},
            "full": 16383 - BLACK,
            "bands": [1.0, 2.5],
            "fractions": [0.76, 0.05],
            "power_sigma": 2.0,
            "chroma_sigma": 1.5,
            "strength": {
                "iso": [160, 3200],
                "nr": [-4, 0, 4],
                "k": [[1.0, 1.5, 2.0], [1.5, 2.0, 3.0]],
            },
        },
        "sharpness": {
            "sigma": 1.2,
            "levels": {
                "-3": _levels(0.3),
                "+0": _levels(1.0),
                "+4": _levels(2.0),
            },
        },
        "clarity": {
            "sigma": 8.0,
            "fine_sigma": 1.0,
            "fine_weight": 0.4,
            "centres": [0.0, 128.0, 255.0],
            "levels": {
                "-5": {"gain": [-0.5, -0.5, -0.5], "offset": 0.0},
                "+5": {"gain": [0.5, 0.5, 0.5], "offset": 0.0},
            },
        },
        "grain": {
            "large_sigma": 1.0,
            "curves": {
                "Weak/Small": curve,
                "Weak/Large": curve,
                "Strong/Small": strong,
                "Strong/Large": strong,
            },
            "acros": {
                "Weak/Small": acros,
                "Weak/Large": acros,
                "Strong/Small": acros,
                "Strong/Large": acros,
            },
            "downscale": {
                "Small": [[0.1, 0.1], [1.0, 1.0]],
                "Large": [[0.1, 0.4], [1.0, 1.0]],
            },
        },
    }


def write_texture(bundle: Path) -> Path:
    """Write the texture calibration next to a bundle."""
    path = bundle.with_suffix(".texture.json")
    path.write_text(json.dumps(texture()))
    return path


def mosaic(
    by_colour: tuple[float, float, float], h: int = RAW_H, w: int = RAW_W
) -> Mosaic:
    """A flat raw frame: each site its colour's level above black."""
    colors = np.tile(XTRANS, (-(-h // 6), -(-w // 6)))[:h, :w]
    data = np.asarray(by_colour, float)[colors] + BLACK
    return Mosaic(data.astype(np.uint16), XTRANS, BLACK)
