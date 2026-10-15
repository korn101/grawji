"""The Recipe row names the saved recipe the open image was shot with."""

from __future__ import annotations

import time
from typing import Any

import pytest

from grawji.recipe import Recipe
from tests.gui_support import pump

pytestmark = pytest.mark.gui

SHOT = Recipe(film_simulation="ClassicChrome", white_balance="Auto", color=3)


class _Entry:
    """The slice of Gtk.Entry the save dialog's response reads."""

    def __init__(self, text: str) -> None:
        self._text = text

    def get_text(self) -> str:
        return self._text


def _subtitle(win: Any) -> str:
    return str(win.recipe_panel.recipe_row.get_subtitle() or "")


def _save(win: Any, name: str, recipe: Recipe) -> None:
    """Save a recipe the way the Save dialog does."""
    win._library._on_save_response(None, "save", _Entry(name), recipe, False)


def test_panel_shows_the_subtitle_and_tooltip() -> None:
    """set_shot_with fills, and clears, the row's subtitle and tooltip."""
    from grawji.views.recipe_panel import RecipePanel

    panel = RecipePanel()
    panel.set_shot_with("Shot with Old Kodak", "tip")
    assert panel.recipe_row.get_subtitle() == "Shot with Old Kodak"
    assert panel.recipe_row.get_tooltip_text() == "tip"
    panel.set_shot_with("")
    assert not panel.recipe_row.get_subtitle()
    assert panel.recipe_row.get_tooltip_text() is None


def test_library_changes_update_the_name(window: Any) -> None:
    """Saving, renaming and deleting the matching recipe show at once."""
    win = window
    win._on_shot_recipe(win._generation, SHOT)
    assert _subtitle(win) == "Shot with Classic Chrome · no saved recipe"

    _save(win, "Old Kodak", SHOT)
    assert _subtitle(win) == "Shot with Old Kodak"

    win._library.rename_recipe("Old Kodak", "C6 Old Kodak")
    assert _subtitle(win) == "Shot with C6 Old Kodak"

    win._library.delete_recipe("C6 Old Kodak")
    assert _subtitle(win) == "Shot with Classic Chrome · no saved recipe"


def test_a_superseded_read_is_ignored(window: Any) -> None:
    """A late read for an image no longer open changes nothing."""
    win = window
    win._on_shot_recipe(win._generation - 1, SHOT)
    assert _subtitle(win) == ""


def test_opening_an_image_reads_its_recipe(
    window: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """The decode thread reads the recipe and the row names it."""
    from grawji.views import window as window_module

    win = window
    _save(win, "Old Kodak", SHOT)
    monkeypatch.setattr(window_module, "_shot_recipe_of", lambda _p: SHOT)
    win._decode_selection(win._generation, str(tmp_path / "x.RAF"))
    deadline = time.monotonic() + 5
    while _subtitle(win) != "Shot with Old Kodak":
        assert time.monotonic() < deadline, _subtitle(win)
        pump()
        time.sleep(0.01)


def test_an_odd_file_only_loses_the_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tag the parser chokes on gives no recipe instead of an error."""
    from grawji.views import window as window_module

    monkeypatch.setattr(
        window_module.imagemeta, "photo_tags", lambda _p: lambda _t: "   "
    )
    assert window_module._shot_recipe_of("odd.RAF") is None
