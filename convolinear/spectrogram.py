"""Spectrogram class for time-frequency signal representations."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _frozen(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return a read-only float64 copy of ``values`` (never aliases the input)."""
    arr = np.asarray(values, dtype=np.float64)
    if arr is values:
        # np.asarray returned the caller's own object - copy so freezing
        # doesn't mutate an array the caller still holds.
        arr = arr.copy()
    arr.flags.writeable = False
    return arr


class Spectrogram:
    """A time-frequency representation of a signal.

    Produced by ``Signal.spectrogram()``. Holds a 2-D array of magnitudes
    indexed by frequency (rows) and time (columns), together with the
    frequency and time axes that label them.

    Magnitudes use the same convention as ``Signal.fft()``: a sinusoid of
    amplitude 1 reads a magnitude of roughly 1 in the frequency bin and time
    frames where it is present.
    """

    def __init__(
        self,
        frequencies: npt.ArrayLike,
        times: npt.ArrayLike,
        magnitudes: npt.ArrayLike,
    ):
        freqs = _frozen(frequencies)
        frames = _frozen(times)
        mags = _frozen(magnitudes)

        if len(freqs) == 0 or len(frames) == 0:
            raise ValueError(
                "Spectrogram must have at least one frequency bin and one time frame; "
                f"got {len(freqs)} frequencies and {len(frames)} frames."
            )

        expected = (len(freqs), len(frames))
        if mags.shape != expected:
            raise ValueError(
                f"magnitudes must have shape (n_frequencies, n_times) = {expected}, "
                f"got {mags.shape}"
            )

        self._frequencies = freqs
        self._times = frames
        self._magnitudes = mags

    @property
    def frequencies(self) -> npt.NDArray[np.float64]:
        """The frequency axis in Hz (read-only)."""
        return self._frequencies

    @property
    def times(self) -> npt.NDArray[np.float64]:
        """The time axis in seconds (read-only)."""
        return self._times

    @property
    def magnitudes(self) -> npt.NDArray[np.float64]:
        """The 2-D magnitude array indexed by frequency (rows) and time (columns), read-only."""
        return self._magnitudes

    def __len__(self) -> int:
        """Number of time frames."""
        return len(self.times)

    def __repr__(self) -> str:
        return (
            f"Spectrogram(frequencies={len(self.frequencies)}, "
            f"frames={len(self.times)}, "
            f"freq_range=({self.frequencies[0]:.1f}, {self.frequencies[-1]:.1f}) Hz, "
            f"duration={self.times[-1]:.3f} s)"
        )

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the magnitude array as ``(n_frequencies, n_times)``."""
        return cast("tuple[int, int]", self.magnitudes.shape)

    def peak_frequency_over_time(self) -> npt.NDArray[np.float64]:
        """The dominant frequency (Hz) in each time frame.

        Returns an array the same length as ``times`` giving, for every frame,
        the frequency with the largest magnitude. Useful for tracking how a
        tone or formant moves over time (e.g. a chirp or a glissando).
        """
        return cast("npt.NDArray[np.float64]", self.frequencies[np.argmax(self.magnitudes, axis=0)])

    def plot(
        self,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        max_freq: float | None = None,
        db_scale: bool = True,
        colorbar: bool = True,
        cmap: str = "magma",
        ax: Axes | None = None,
    ) -> Axes:
        """Plot the spectrogram as a heatmap. Returns the matplotlib axis.

        Args:
            title:     Plot title.
            xlabel:    X-axis label (defaults to ``"Time (s)"``).
            ylabel:    Y-axis label (defaults to ``"Frequency (Hz)"``).
            max_freq:  If set, limit the frequency axis to this value in Hz.
            db_scale:  Show magnitude in decibels (default). When ``False``,
                       a linear magnitude scale is used.
            colorbar:  Draw a colour bar alongside the plot.
            cmap:      matplotlib colormap name.
            ax:        Existing matplotlib ``Axes`` to draw on.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        from .spectrum import _limit_to_max_freq

        freqs, mags = _limit_to_max_freq(self.frequencies, self.magnitudes, max_freq)

        if db_scale:
            # Floor tiny values so log10 stays finite, then convert to dB.
            floor = np.max(mags) * 1e-6 if np.max(mags) > 0 else 1e-12
            values = 20 * np.log10(np.maximum(mags, floor))
            cbar_label = "Magnitude (dB)"
        else:
            values = mags
            cbar_label = "Magnitude"

        mesh = ax.pcolormesh(self.times, freqs, values, shading="auto", cmap=cmap)
        ax.set_xlabel(xlabel or "Time (s)")
        ax.set_ylabel(ylabel or "Frequency (Hz)")
        ax.set_title(title or "Spectrogram")
        if colorbar:
            ax.figure.colorbar(mesh, ax=ax, label=cbar_label)
        return ax
