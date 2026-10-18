"""Approximate recipe renders while no camera is connected.

An image qualifies when a calibration bundle of its body is installed.
With the body's texture calibration too, its raw data is developed once
at a third of the output size and each recipe is applied to that, grain,
clarity, sharpness and noise reduction included. Otherwise, when it was
shot with a colour film simulation, its built-in JPEG is put through the
recipe's look table. Either runs off the main thread, and the result is
shown as an approximation.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import gi

gi.require_version("GdkPixbuf", "2.0")

import numpy as np
from gi.repository import GdkPixbuf, GLib
from numpy.typing import NDArray

from grawji import mainloop
from grawji.imaging import imagemeta
from grawji.imaging.approximate import (
    Calibration,
    Shot,
    apply_table,
    calibration_for,
)
from grawji.imaging.develop import (
    PUSH,
    Geometry,
    develop,
    gain_map,
    raw_info,
    read_raf,
)
from grawji.imaging.pixbufs import orient_exif, rgb_array
from grawji.imaging.texture import clarity, denoise, grain, sharpen
from grawji.recipe import Recipe

# Previews are developed at a third of the camera's output size.
PREVIEW_FACTOR = 3
# One grain pattern per image, so it does not crawl as settings change.
_GRAIN_SEED = 20261018
# Noise reduction assumes the base ISO when a file does not record one.
_BASE_ISO = 160


@dataclass(frozen=True)
class RawSource:
    """A RAF whose raw data is developed for approximations.

    Attributes:
        path: The RAF.
        iso: The ISO it was shot at, which sets its noise.
        orientation: Its EXIF orientation (the raw data is as the sensor
            saw it).
    """

    path: str
    iso: float
    orientation: int = 1


@dataclass(frozen=True)
class Prepared:
    """What approximating one image needs, read off the main thread.

    Attributes:
        calibration: The body's tables.
        shot: The recipe and white balance the photo was taken with.
        model: The body, for its capabilities.
        raw: Where its raw data comes from, or None to approximate from
            the built-in JPEG.
    """

    calibration: Calibration
    shot: Shot
    model: str
    raw: RawSource | None = None


def prepare(raf_path: str, shot: Recipe | None) -> Prepared | None:
    """The approximation inputs of a RAF, or None if it does not qualify.

    Reads metadata and may load a bundle, so it runs on a worker thread.
    """
    if shot is None:
        return None
    model = imagemeta.camera_model(raf_path)
    calibration = calibration_for(model)
    if model is None or calibration is None:
        return None
    raw = _raw_prepared(raf_path, shot, calibration, model)
    if raw is not None:
        return raw
    if not calibration.can_approximate(shot, shot):
        return None
    levels = imagemeta.fuji_auto_wb_levels(raf_path)
    return Prepared(calibration, calibration.shot(shot, levels), model)


def _raw_prepared(
    raf_path: str, shot: Recipe, calibration: Calibration, model: str
) -> Prepared | None:
    """The inputs for developing a RAF's raw data, if it can be."""
    texture = calibration.texture
    if texture is None or not calibration.can_approximate(
        shot, shot, raw=True
    ):
        return None
    try:
        info = raw_info(raf_path)
        geometry = Geometry.from_meta(texture["develop"])
    except (OSError, ValueError, KeyError):
        return None
    if not geometry.fits(info.rows, info.columns) or min(info.as_shot) <= 0:
        return None
    source = RawSource(
        raf_path,
        float(imagemeta.iso_speed(raf_path) or _BASE_ISO),
        imagemeta.exif_orientation(raf_path),
    )
    return Prepared(
        calibration, calibration.raw_shot(shot, info.as_shot), model, source
    )


@dataclass
class _Developed:
    """An image's raw data developed to preview size.

    Attributes:
        source: The RAF it came from.
        scene: Scene levels, rows x columns x RGB.
        gain: The lens gain over the same frame.
        denoised: The scene after the last noise reduction level used.
    """

    source: RawSource
    scene: NDArray[np.float32]
    gain: NDArray[np.float32]
    denoised: dict[int, NDArray[np.float32]] = field(default_factory=dict)


def _to_pixbuf(rgb: NDArray[np.uint8]) -> Any:
    height, width, _ = rgb.shape
    return GdkPixbuf.Pixbuf.new_from_bytes(
        GLib.Bytes.new(rgb.tobytes()),
        GdkPixbuf.Colorspace.RGB,
        False,
        8,
        width,
        height,
        width * 3,
    )


class Approximation:
    """The open image's approximation source and its renders."""

    def __init__(
        self,
        *,
        on_done: Callable[[int, int, Any], None],
        on_error: Callable[[int, int, str], None],
    ) -> None:
        """Set up with no image.

        Args:
            on_done: Called on the main loop with the generation, the
                render sequence and the approximated pixbuf.
            on_error: Called on the main loop with the generation, the
                sequence and a message when a render fails.
        """
        self._on_done = on_done
        self._on_error = on_error
        self._prepared: Prepared | None = None
        self._pixels: NDArray[np.uint8] | None = None
        self._developed: _Developed | None = None
        self._develop_lock = threading.Lock()
        self.seq = 0

    @property
    def ready(self) -> bool:
        """Whether the open image can be approximated."""
        return self._prepared is not None and self._pixels is not None

    @property
    def prepared(self) -> Prepared | None:
        """The open image's approximation inputs, if any."""
        return self._prepared

    def clear(self) -> None:
        """Forget the image (a new selection, or a camera took over)."""
        self._prepared = None
        self._pixels = None
        self._developed = None
        self.seq += 1

    def load(self, prepared: Prepared | None, pixbuf: Any) -> bool:
        """Take the image's inputs and its decoded built-in JPEG."""
        self.clear()
        if prepared is None or pixbuf is None:
            return False
        self._prepared = prepared
        self._pixels = rgb_array(pixbuf)
        return True

    def render(self, generation: int, recipe: Recipe) -> None:
        """Approximate recipe on the open image, off the main thread."""
        prepared, pixels = self._prepared, self._pixels
        if prepared is None or pixels is None:
            return
        self.seq += 1
        seq = self.seq

        def work() -> None:
            try:
                if prepared.raw is None:
                    table = prepared.calibration.look_table(
                        prepared.shot, recipe
                    )
                    pixbuf = _to_pixbuf(apply_table(pixels, table))
                else:
                    rgb = self._render_raw(prepared, prepared.raw, recipe, seq)
                    if rgb is None:
                        return  # superseded
                    pixbuf = orient_exif(
                        _to_pixbuf(rgb), prepared.raw.orientation
                    )
            except (KeyError, ValueError, OSError) as exc:
                mainloop.call(self._on_error, generation, seq, str(exc))
                return
            mainloop.call(self._on_done, generation, seq, pixbuf)

        threading.Thread(
            target=work, name="grawji-approximate", daemon=True
        ).start()

    def _develop(self, prepared: Prepared, source: RawSource) -> _Developed:
        """The image's developed raw data, made by the first render."""
        with self._develop_lock:
            developed = self._developed
            if developed is not None and developed.source is source:
                return developed
            texture = prepared.calibration.texture
            levels = prepared.shot.levels
            if texture is None or levels is None:
                raise ValueError("no texture calibration for the raw data")
            geometry = Geometry.from_meta(texture["develop"])
            scene = develop(
                read_raf(source.path),
                levels,
                geometry,
                prepared.shot.recipe.dynamic_range,
                PREVIEW_FACTOR,
            )
            height, width = scene.shape[:2]
            developed = _Developed(
                source, scene, gain_map(geometry, height, width)
            )
            if self._prepared is prepared:
                self._developed = developed
            return developed

    def _render_raw(
        self, prepared: Prepared, source: RawSource, recipe: Recipe, seq: int
    ) -> NDArray[np.uint8] | None:
        """Recipe on the developed raw data, or None once superseded."""
        developed = self._develop(prepared, source)
        if seq != self.seq:
            return None
        cal, shot = prepared.calibration, prepared.shot
        texture = cal.texture
        if texture is None or shot.levels is None:
            raise ValueError("no texture calibration for the raw data")
        scale = 1 / PREVIEW_FACTOR
        level = recipe.noise_reduction
        scene = developed.denoised.get(level)
        if scene is None:
            scene = denoise(
                developed.scene,
                level,
                texture["noise"],
                iso=source.iso,
                push=PUSH[shot.recipe.dynamic_range],
                levels=shot.levels,
                gain=developed.gain,
                scale=scale,
            )
            # Keep only the latest level: a slider drag passes through many.
            developed.denoised = {level: scene}
        table = cal.scene_table(shot, recipe)
        img = cal.apply_scene_table(scene, table) * 255
        img = clarity(img, recipe.clarity, texture["clarity"], scale)
        img = sharpen(img, recipe.sharpness, texture["sharpness"], scale)
        img = grain(
            img,
            recipe.grain,
            recipe.grain_size,
            recipe.film_simulation,
            texture["grain"],
            rng=np.random.default_rng(_GRAIN_SEED),
            scale=scale,
        )
        return np.clip(np.rint(img), 0, 255).astype(np.uint8)


__all__ = [
    "PREVIEW_FACTOR",
    "Approximation",
    "Prepared",
    "RawSource",
    "prepare",
]
