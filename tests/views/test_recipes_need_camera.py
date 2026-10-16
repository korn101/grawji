"""Recipes are applied only when the camera can develop them."""

from __future__ import annotations

from typing import Any

import pytest

from grawji.camera import camera_info
from grawji.camera.core import CameraSession
from grawji.recipe import Recipe
from grawji.views import window as window_module

pytestmark = pytest.mark.gui

NAME = "Astia look"
JPEG_STATUS = "Recipes develop the RAW. Switch to it with j."


class FakeWorker:
    """Takes renders without a camera."""

    def render(self, *args: Any, **kwargs: Any) -> None:
        """Renders are not looked at here."""

    def stop(self, **kwargs: Any) -> None:
        """Nothing to stop."""


@pytest.fixture
def win(window: Any) -> Any:
    """The window, with no camera, and one saved recipe."""
    window._recipe_library.add(NAME, Recipe(film_simulation="Astia"))
    return window


def _status(win: Any) -> str:
    return str(win.preview_view.status.get_label())


def test_picking_a_recipe_asks_for_the_camera(
    win: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing changes, and no unsaved-edits prompt comes first."""
    prompts: list[Any] = []
    monkeypatch.setattr(win, "_confirm_recipe_discard", prompts.append)
    label = win.recipe_panel.active_label
    win._on_apply_recipe(None, NAME)
    assert _status(win) == window_module._RECIPES_NEED_CAMERA
    assert prompts == []
    assert win.recipe_panel.active_label == label
    assert win._settings.open_recipe != NAME


def test_a_recipe_key_asks_for_the_camera(win: Any) -> None:
    """The number key is taken, and the recipe is not applied."""
    from gi.repository import Gdk

    win._recipe_library.set_hotkey(NAME, 1)
    assert win._on_recipe_key(Gdk.KEY_1, Gdk.ModifierType(0))
    assert _status(win) == window_module._RECIPES_NEED_CAMERA
    assert win.recipe_panel.active_label != NAME


def test_try_recipes_asks_for_the_camera(win: Any) -> None:
    """Try Recipes names the camera, not a missing image."""
    win._on_try_recipes()
    assert _status(win) == window_module._RECIPES_NEED_CAMERA


def test_reset_asks_for_the_camera(win: Any) -> None:
    """Resetting to the default recipe is held back too."""
    win.recipe_panel.set_active(Recipe(film_simulation="Astia"), NAME)
    win._reset_recipe()
    assert _status(win) == window_module._RECIPES_NEED_CAMERA
    assert win.recipe_panel.active_label == NAME


def test_a_camera_jpeg_asks_for_the_raw(
    win: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the camera JPEG shown, the picker points to the RAW."""
    monkeypatch.setattr(camera_info, "detect_camera", lambda: "X100V")
    monkeypatch.setattr(win, "_camera_file_of", lambda _path: "a.JPG")
    win._on_apply_recipe(None, NAME)
    assert _status(win) == JPEG_STATUS
    assert win.recipe_panel.active_label != NAME


def test_a_connected_camera_takes_the_recipe(
    win: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A camera on USB still opening the image gets the recipe."""
    monkeypatch.setattr(camera_info, "detect_camera", lambda: "X100V")
    win._on_apply_recipe(None, NAME)
    assert win.recipe_panel.active_label == NAME


def test_an_open_session_takes_the_recipe(
    win: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the image open on the camera, the recipe applies."""
    win._worker.stop(close_session=False)
    monkeypatch.setattr(win, "_worker", FakeWorker())
    monkeypatch.setattr(CameraSession, "is_open", property(lambda s: True))
    win._on_apply_recipe(None, NAME)
    assert win.recipe_panel.active_label == NAME
    assert _status(win) != window_module._RECIPES_NEED_CAMERA
