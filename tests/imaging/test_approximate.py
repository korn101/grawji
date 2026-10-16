"""Approximating recipes from a body's calibration tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from grawji.imaging.approximate import (
    Calibration,
    apply_table,
    bundle_path,
    calibration_for,
)
from grawji.recipe import Recipe
from tests.approx_support import STOP, write_bundle

# Mid tones: far from clipping at either end after a stop either way.
MID = np.full((4, 6, 3), 128, dtype=np.uint8)
AUTO_LEVELS = (100, 180, 180)  # G, R, B: the bundle's carrier levels


@pytest.fixture
def cal(tmp_path: Path) -> Calibration:
    """The test bundle."""
    return Calibration(write_bundle(tmp_path / "X100V.npz"))


def _run(cal: Calibration, shot: Recipe, target: Recipe) -> np.ndarray:
    table = cal.look_table(cal.shot(shot, AUTO_LEVELS), target)
    return apply_table(MID, table).astype(float)


def test_the_shot_recipe_gives_the_image_back(cal: Calibration) -> None:
    """Undoing a shot and redoing its own recipe changes nothing."""
    ramp = np.linspace(0, 255, 4 * 6 * 3).astype(np.uint8).reshape(4, 6, 3)
    shot = Recipe(film_simulation="Provia")
    out = apply_table(ramp, cal.look_table(cal.shot(shot, None), shot))
    assert np.abs(out.astype(int) - ramp).max() <= 1


def test_exposure_counts_from_the_shot(cal: Calibration) -> None:
    """The built-in JPEG has the shot's EV already."""
    plus = _run(cal, Recipe(), Recipe(exposure=1.0))
    assert plus.mean() == pytest.approx(128 + STOP * 255, abs=1)
    same = _run(cal, Recipe(exposure=1.0), Recipe(exposure=1.0))
    assert same.mean() == pytest.approx(128, abs=1)


def test_a_dynamic_range_above_the_shot_is_capped(cal: Calibration) -> None:
    """A DR100 shot cannot render DR400, as on the camera."""
    capped = _run(cal, Recipe(), Recipe(dynamic_range="DR400"))
    assert capped.mean() == pytest.approx(128, abs=1)
    used = _run(
        cal, Recipe(dynamic_range="DR400"), Recipe(dynamic_range="DR400")
    )
    assert used.mean() == pytest.approx(0.25 * 255, abs=1)


def test_a_fixed_white_balance_rescales_from_the_shot(
    cal: Calibration,
) -> None:
    """Shade doubles red and halves blue against the shot's levels."""
    out = _run(
        cal, Recipe(white_balance="Auto"), Recipe(white_balance="Shade")
    )
    r, g, b = out.reshape(-1, 3).mean(0)
    assert r == pytest.approx(128 + STOP * 255, abs=1)
    assert g == pytest.approx(128, abs=1)
    assert b == pytest.approx(128 - STOP * 255, abs=1)
    kelvin = _run(
        cal,
        Recipe(),
        Recipe(white_balance="Temperature", color_temp=10000),
    )
    assert kelvin.reshape(-1, 3).mean(0)[0] == pytest.approx(r, abs=1)


def test_auto_white_balance_keeps_the_shot(cal: Calibration) -> None:
    """The camera's Auto on a RAF reuses what the shot applied."""
    out = _run(
        cal, Recipe(white_balance="Daylight"), Recipe(white_balance="Auto")
    )
    assert np.abs(out - 128).max() <= 1


def test_unknown_shot_levels_leave_white_balance(cal: Calibration) -> None:
    """Without the shot's levels a fixed WB cannot be placed."""
    shot = cal.shot(Recipe(white_balance="Auto"), None)
    assert shot.levels is None
    out = apply_table(
        MID, cal.look_table(shot, Recipe(white_balance="Shade"))
    ).astype(float)
    assert np.abs(out - 128).max() <= 1


def test_wb_shift_steps_interpolate(cal: Calibration) -> None:
    """Measured steps are exact, the ones between are log-interpolated."""
    np.testing.assert_allclose(cal.shift_gain(3, -3), [2.0, 1.0, 0.5])
    assert cal.shift_gain(1, 0)[0] == pytest.approx(2 ** (1 / 3))
    out = _run(cal, Recipe(), Recipe(wb_shift_r=3))
    assert out.reshape(-1, 3).mean(0)[0] == pytest.approx(
        128 + STOP * 255, abs=1
    )


def test_the_shot_settings_are_undone(cal: Calibration) -> None:
    """A shot's colour +2 comes off before the target's goes on."""
    shot = Recipe(color=2)
    same = _run(cal, shot, shot)
    assert np.abs(same - 128).max() <= 1
    # 128 was 0.5 + x / 2 for x ~ 0.004: undoing leaves nearly black.
    plain = _run(cal, shot, Recipe())
    assert plain.max() <= 3


def test_a_bw_shot_cannot_be_approximated(cal: Calibration) -> None:
    """A B&W JPEG has no colour to take back."""
    assert not cal.can_approximate(
        Recipe(film_simulation="Acros"), Recipe(film_simulation="Provia")
    )
    assert cal.can_approximate(
        Recipe(film_simulation="Provia"), Recipe(film_simulation="Acros")
    )
    assert not cal.can_approximate(
        Recipe(film_simulation="Provia"), Recipe(film_simulation="Eterna")
    )


def test_a_dr_auto_shot_cannot_be_approximated(cal: Calibration) -> None:
    """The metadata does not say which DR table would undo it."""
    assert not cal.can_approximate(Recipe(dynamic_range="Auto"), Recipe())


def test_the_installed_bundle_is_found() -> None:
    """A body's bundle is read from grawji's data directory."""
    assert calibration_for("X100V") is None
    assert calibration_for(None) is None
    write_bundle(bundle_path("X100V"))
    assert calibration_for("X100V") is not None
    assert calibration_for("X-T5") is None
