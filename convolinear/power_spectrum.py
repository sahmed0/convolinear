"""PowerSpectrum class for power spectral density estimates."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy import signal as scipy_signal

from .spectrum import _limit_to_max_freq

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _frozen(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return a read-only float64 copy of ``values``."""
    arr = np.asarray(values, dtype=np.float64)
    if arr is values:
        # np.asarray returned the caller's own object - copy so freezing
        # doesn't mutate an array the caller still holds.
        arr = arr.copy()
    arr.flags.writeable = False
    return arr


class PowerSpectrum:
    """A power spectral density estimate (power per Hz vs frequency).

    Produced by :meth:`Signal.psd` (Welch's method). Unlike :class:`Spectrum`
    this is a statistical *estimate*: it carries no phase and cannot be
    inverted back to a signal. Values are densities (units^2 / Hz).

    The stored ``frequencies`` and ``power`` arrays are read-only.
    """

    def __init__(self, frequencies: npt.ArrayLike, power: npt.ArrayLike):
        """Create a PowerSpectrum from a frequency axis and density values.

        Args:
            frequencies: The 1-D frequency axis in Hz.
            power:       The power spectral density at each frequency (units^2
                         per Hz), same shape as ``frequencies``.

        Raises:
            ValueError: if the two arrays differ in shape or are empty.
        """
        freqs = _frozen(frequencies)
        pwr = _frozen(power)

        if freqs.shape != pwr.shape:
            raise ValueError(
                f"frequencies and power must have the same shape, got {freqs.shape} and {pwr.shape}"
            )
        if len(freqs) == 0:
            raise ValueError("PowerSpectrum must contain at least one bin.")

        self._frequencies = freqs
        self._power = pwr

    def __len__(self) -> int:
        return len(self._frequencies)

    def __repr__(self) -> str:
        return (
            f"PowerSpectrum(bins={len(self._frequencies)}, "
            f"freq_range=({self._frequencies[0]:.1f}, {self._frequencies[-1]:.1f}) Hz)"
        )

    @property
    def frequencies(self) -> npt.NDArray[np.float64]:
        """The frequency axis in Hz (read-only)."""
        return self._frequencies

    @property
    def power(self) -> npt.NDArray[np.float64]:
        """The power spectral density at each frequency (read-only)."""
        return self._power

    @property
    def peak_frequency(self) -> float:
        """The frequency with the highest power density (dominant frequency)."""
        return float(self._frequencies[np.argmax(self._power)])

    @property
    def peak_power(self) -> float:
        """The power density at the peak frequency."""
        return float(np.max(self._power))

    def top_n(self, n: int = 5, min_prominence: float | None = None) -> list[tuple[float, float]]:
        """Return up to ``n`` power-density peaks as (frequency, power) pairs.

        Peaks are true local maxima found with :func:`scipy.signal.find_peaks`
        (not the ``n`` largest bins). Results are ordered by descending power.
        The list may contain fewer than ``n`` entries - or be empty - when the
        estimate has fewer qualifying peaks. Bins at the very edges (DC and
        Nyquist) cannot qualify as peaks.

        Args:
            n:              Maximum number of peaks to return.
            min_prominence: Optional prominence threshold, in power units,
                            forwarded to ``find_peaks``. Use it to suppress
                            noise peaks.

        Raises:
            ValueError: if ``n`` is less than 1.
        """
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}.")
        peak_idx, _ = scipy_signal.find_peaks(self._power, prominence=min_prominence)
        order = np.argsort(self._power[peak_idx])[::-1][:n]
        top = peak_idx[order]
        return [(float(self._frequencies[i]), float(self._power[i])) for i in top]

    def in_range(self, low: float, high: float) -> PowerSpectrum:
        """Return the sub-band ``[low, high]`` Hz as a new PowerSpectrum.

        Unlike :meth:`Spectrum.in_range`, this is a plain subset: the returned
        estimate keeps only the in-band bins (a PSD carries no phase and is not
        invertible, so there is no reason to preserve the full axis).

        Raises:
            ValueError: if ``low > high`` or no bin falls inside the band.
        """
        if low > high:
            raise ValueError(f"low ({low}) must not exceed high ({high}).")
        mask = (self._frequencies >= low) & (self._frequencies <= high)
        if not mask.any():
            spacing = (
                self._frequencies[1] - self._frequencies[0] if len(self._frequencies) > 1 else 0
            )
            raise ValueError(
                f"No frequency bins in [{low}, {high}] Hz. This estimate spans "
                f"{self._frequencies[0]:g}-{self._frequencies[-1]:g} Hz with "
                f"{spacing:g} Hz spacing."
            )
        return PowerSpectrum(self._frequencies[mask], self._power[mask])

    def plot(
        self,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        log_scale: bool = True,
        max_freq: float | None = None,
        ax: Axes | None = None,
    ) -> Axes:
        """Plot the power spectral density. Returns the matplotlib axis.

        The y-axis is logarithmic by default, since PSDs commonly span several
        decades; pass ``log_scale=False`` for a linear scale.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 3))

        freqs, pwr = _limit_to_max_freq(self._frequencies, self._power, max_freq)

        ax.plot(freqs, pwr, linewidth=0.8)
        ax.set_xlabel(xlabel or "Frequency (Hz)")
        ax.set_ylabel(ylabel or "Power spectral density")
        ax.set_title(title or "Power Spectral Density")
        if log_scale:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        return ax
