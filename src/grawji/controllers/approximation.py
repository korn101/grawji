"""Approximate recipe renders while no camera is connected.

An image qualifies when a calibration bundle of its body is installed and
it was shot with a colour film simulation. Its built-in JPEG is then put
through the recipe's look table off the main thread, and the result is
shown as an approximation.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
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
from grawji.imaging.pixbufs import rgb_array
from grawji.recipe import Recipe


@dataclass(frozen=True)
class Prepared:
    """What approximating one image needs, read off the main thread.

    Attributes:
        calibration: The body's tables.
        shot: The recipe and white balance the photo was taken with.
        model: The body, for its capabilities.
    """

    calibration: Calibration
    shot: Shot
    model: str


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
    if not calibration.can_approximate(shot, shot):
        return None
    levels = imagemeta.fuji_auto_wb_levels(raf_path)
    return Prepared(calibration, calibration.shot(shot, levels), model)


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
                table = prepared.calibration.look_table(prepared.shot, recipe)
                pixbuf = _to_pixbuf(apply_table(pixels, table))
            except (KeyError, ValueError) as exc:
                mainloop.call(self._on_error, generation, seq, str(exc))
                return
            mainloop.call(self._on_done, generation, seq, pixbuf)

        threading.Thread(
            target=work, name="grawji-approximate", daemon=True
        ).start()


__all__ = ["Approximation", "Prepared", "prepare"]
