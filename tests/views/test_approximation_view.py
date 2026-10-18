"""Approximated recipes in the window while no camera is connected."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from grawji.camera.core import CameraSession
from grawji.controllers import approximation
from grawji.controllers.approximation import Prepared, RawSource
from grawji.imaging.approximate import Calibration
from grawji.recipe import Recipe
from grawji.views import window as window_module
from tests.approx_support import mosaic, write_bundle, write_texture
from tests.gui_support import pump

pytestmark = pytest.mark.gui

NAME = "Colour plus two"
SHOT = Recipe(film_simulation="Provia")


def _jpeg() -> bytes:
    from gi.repository import GdkPixbuf

    pixbuf = GdkPixbuf.Pixbuf.new(GdkPixbuf.Colorspace.RGB, False, 8, 12, 8)
    pixbuf.fill(0x808080FF)
    ok, data = pixbuf.save_to_bufferv("jpeg", ["quality"], ["100"])
    assert ok
    return bytes(data)


@pytest.fixture
def win(window: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    """The window with no camera, whose RAFs have a calibrated body."""
    cal = Calibration(write_bundle(tmp_path / "X100V.npz"))
    prepared = Prepared(cal, cal.shot(SHOT, (100, 180, 180)), "X100V")
    monkeypatch.setattr(window_module.raf, "embedded_jpeg", lambda _p: _jpeg())
    monkeypatch.setattr(window_module, "_shot_recipe_of", lambda _p: SHOT)
    monkeypatch.setattr(
        window_module, "_offline_inputs", lambda _p, _s: prepared
    )
    window._recipe_library.add(NAME, Recipe(color=2))
    (tmp_path / "a.RAF").write_bytes(b"not a real raf")
    window.test_raf = str(tmp_path / "a.RAF")
    return window


def _wait(condition: Any, what: str) -> None:
    deadline = time.monotonic() + 5
    while not condition():
        assert time.monotonic() < deadline, what
        pump()
        time.sleep(0.01)


def _status(win: Any) -> str:
    return str(win.preview_view.status.get_label())


def _open(win: Any) -> None:
    win._on_raf_selected(win.test_raf)
    _wait(win._offline_ready, "the image never became approximable")


def test_an_image_opens_ready_to_approximate(win: Any) -> None:
    """Its own recipe shows the built-in JPEG, the controls are live."""
    _open(win)
    assert _status(win) == window_module._APPROXIMATING
    assert win.recipe_panel.film_row.get_sensitive()
    assert not win.preview_view.approximation_badge.get_visible()


def test_a_picked_recipe_is_approximated(win: Any) -> None:
    """The recipe applies and its approximation shows, labelled."""
    _open(win)
    win._on_apply_recipe(None, NAME)
    assert win.recipe_panel.active_label == NAME
    badge = win.preview_view.approximation_badge
    _wait(badge.get_visible, "no approximation was shown")
    assert _status(win) == window_module._APPROXIMATION
    pixels = win.preview_view.source_pixbuf.get_pixels()
    # The test bundle's colour +2 halves the distance to white.
    assert abs(pixels[0] - 0xBF) <= 2


def test_a_camera_takes_over_from_the_approximation(
    win: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A late approximation is dropped once the camera renders."""
    _open(win)
    win._offline.render(win._generation, Recipe(exposure=1.0))
    monkeypatch.setattr(CameraSession, "is_open", property(lambda s: True))
    seq = win._offline.seq
    _wait(lambda: win._offline.seq == seq, "")
    for _ in range(50):
        pump()
        time.sleep(0.01)
    assert not win.preview_view.approximation_badge.get_visible()


def test_without_a_bundle_recipes_ask_for_the_camera(
    win: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An uncalibrated body keeps the camera-only behaviour."""
    monkeypatch.setattr(window_module, "_offline_inputs", lambda _p, _s: None)
    win._on_raf_selected(win.test_raf)
    _wait(
        lambda: (
            _status(win) == window_module._NO_CAMERA
            and win.preview_view.source_pixbuf is not None
        ),
        "the image never loaded",
    )
    for _ in range(20):
        pump()
    win._on_apply_recipe(None, NAME)
    assert _status(win) == window_module._RECIPES_NEED_CAMERA
    assert not win._offline_ready()


def test_a_connected_camera_is_not_preempted(
    win: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a camera on USB the image waits for it, not approximating."""
    from grawji.camera import camera_info

    opened: list[str] = []
    monkeypatch.setattr(camera_info, "detect_camera", lambda: "X100V")
    monkeypatch.setattr(
        win._worker, "open", lambda path, **_kw: opened.append(path)
    )
    win._on_raf_selected(win.test_raf)
    _wait(lambda: opened and win.preview_view.source_pixbuf, "never loaded")
    for _ in range(20):
        pump()
    assert not win._offline_ready()


def test_an_edit_is_approximated(win: Any) -> None:
    """Changing a control re-renders the approximation."""
    _open(win)
    panel = win.recipe_panel
    panel.film_row.set_selected(panel._film_sims.index("Velvia"))
    _wait(
        win.preview_view.approximation_badge.get_visible,
        "the edit was not approximated",
    )


def test_a_raw_development_is_labelled_as_one(
    win: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With a texture calibration the badge says the raw was developed."""
    bundle = write_bundle(tmp_path / "raw" / "X100V.npz")
    write_texture(bundle)
    cal = Calibration(bundle)
    prepared = Prepared(
        cal,
        cal.raw_shot(SHOT, (100.0, 180.0, 180.0)),
        "X100V",
        RawSource(win.test_raf, 800.0),
    )
    monkeypatch.setattr(
        window_module, "_offline_inputs", lambda _p, _s: prepared
    )
    monkeypatch.setattr(
        approximation, "read_raf", lambda _p: mosaic((300, 500, 250))
    )
    _open(win)
    win._on_apply_recipe(None, NAME)
    badge = win.preview_view.approximation_badge
    _wait(badge.get_visible, "no approximation was shown")
    assert "raw data" in badge.get_tooltip_text()
    assert win.preview_view.source_pixbuf.get_width() == 36
