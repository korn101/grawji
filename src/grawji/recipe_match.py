"""Which saved recipe an image was shot with, judged by its settings.

Cameras do not record the custom bank (C1-C7) a shot used, only the
settings it applied, so the saved recipe with the same look names it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from grawji.recipe import Recipe
from grawji.recipe_dedup import canonical_look
from grawji.recipe_text import film_sim_label

_AS_SHOT = "AsShot"


def same_look(shot: Recipe, saved: Recipe) -> bool:
    """Whether a shot's recorded settings are those of a saved recipe.

    A saved recipe with "AsShot" white balance keeps the shot's own mode
    and temperature, so it matches any of them; its shift still applies.
    """
    if saved.white_balance == _AS_SHOT:
        shot = replace(shot, white_balance=_AS_SHOT)
    return canonical_look(shot) == canonical_look(saved)


def matching_recipe(
    shot: Recipe, saved: Iterable[tuple[str, Recipe]]
) -> str | None:
    """The saved recipe with the shot's look, first in library order.

    A recipe with the shot's exact settings wins over one that matches
    only through its "AsShot" white balance.
    """
    candidates = list(saved)
    look = canonical_look(shot)
    exact = (name for name, r in candidates if canonical_look(r) == look)
    loose = (name for name, r in candidates if same_look(shot, r))
    return next(exact, None) or next(loose, None)


def shot_with(shot: Recipe | None, match: str | None) -> tuple[str, str]:
    """The subtitle and tooltip saying what an image was shot with.

    Both are empty when the image records no Fujifilm recipe.
    """
    if shot is None:
        return "", ""
    if match is not None:
        return (
            f"Shot with {match}",
            "The camera recorded the settings of this saved recipe.",
        )
    film = film_sim_label(shot.film_simulation)
    return (
        f"Shot with {film} · no saved recipe",
        "None of your saved recipes has this image's settings.",
    )
