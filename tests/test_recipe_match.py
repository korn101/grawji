"""Tests for naming the saved recipe an image was shot with."""

from dataclasses import replace

from grawji.recipe import Recipe
from grawji.recipe_match import matching_recipe, same_look, shot_with

OLD_KODAK = Recipe(
    film_simulation="ClassicChrome",
    white_balance="Auto",
    dynamic_range="DR400",
    grain="Strong",
    color_chrome="Strong",
    color_chrome_blue="Strong",
    color=3,
    sharpness=-2,
    noise_reduction=-4,
    clarity=-2,
    wb_shift_r=1,
    wb_shift_b=-6,
)


def test_a_shot_matches_the_recipe_it_was_taken_with():
    """Identical settings name the saved recipe."""
    library = [("Provia", Recipe()), ("Old Kodak", OLD_KODAK)]
    assert matching_recipe(OLD_KODAK, library) == "Old Kodak"


def test_exposure_and_origin_do_not_matter():
    """Per-image exposure and the authoring body are not the look."""
    shot = replace(OLD_KODAK, exposure=-0.7, origin_body="X100V")
    assert same_look(shot, OLD_KODAK)


def test_one_different_setting_is_no_match():
    """Only exact settings count, so a tweak on the camera is no match."""
    shot = replace(OLD_KODAK, color=2)
    assert matching_recipe(shot, [("Old Kodak", OLD_KODAK)]) is None


def test_as_shot_white_balance_matches_any_mode():
    """A saved AsShot recipe keeps the shot's own white balance."""
    saved = replace(OLD_KODAK, white_balance="AsShot")
    shot = replace(OLD_KODAK, white_balance="Temperature", color_temp=5200)
    assert same_look(shot, saved)


def test_as_shot_still_compares_the_shift():
    """The white balance shift applies even to AsShot recipes."""
    saved = replace(OLD_KODAK, white_balance="AsShot")
    assert not same_look(replace(OLD_KODAK, wb_shift_b=-5), saved)


def test_a_specific_mode_must_agree():
    """A saved Daylight recipe does not match an Auto shot."""
    saved = replace(OLD_KODAK, white_balance="Daylight")
    assert not same_look(OLD_KODAK, saved)


def test_first_match_in_library_order_wins():
    """Duplicates in the library resolve to the first listed."""
    library = [("C6", OLD_KODAK), ("Old Kodak", OLD_KODAK)]
    assert matching_recipe(OLD_KODAK, library) == "C6"


def test_exact_settings_win_over_as_shot():
    """A 5200 K recipe beats an earlier AsShot one with the same rest."""
    kelvin = replace(OLD_KODAK, white_balance="Temperature", color_temp=5200)
    library = [
        ("Portra 400", replace(OLD_KODAK, white_balance="AsShot")),
        ("Portra 400 v2", kelvin),
    ]
    assert matching_recipe(kelvin, library) == "Portra 400 v2"
    assert matching_recipe(OLD_KODAK, library) == "Portra 400"


def test_labels():
    """Matched, unmatched and unreadable images read differently."""
    assert shot_with(OLD_KODAK, "Old Kodak")[0] == "Shot with Old Kodak"
    assert shot_with(OLD_KODAK, None)[0] == (
        "Shot with Classic Chrome · no saved recipe"
    )
    assert shot_with(None, None) == ("", "")
