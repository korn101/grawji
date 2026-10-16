"""A tiny calibration bundle whose tables are easy to reason about.

Forward and inverse tables are identities in the log-encoded scene, so
undoing a shot and redoing the same recipe gives the image back, and an
exposure step moves every value by a known amount.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

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
