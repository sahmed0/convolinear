"""Internal helpers shared by the spectrum, power spectrum and spectrogram modules.

Not re-exported from ``convolinear``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy import signal as scipy_signal


def frozen(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return a read-only float64 copy of ``values`` (never aliases the input)."""
    arr = np.asarray(values, dtype=np.float64)
    if arr is values:
        # np.asarray returned the caller's own object - copy so freezing
        # doesn't mutate an array the caller still holds.
        arr = arr.copy()
    arr.flags.writeable = False
    return arr


def limit_to_max_freq(
    frequencies: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    max_freq: float | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Restrict a frequency axis (and its values) to ``<= max_freq``.

    Shared by :meth:`Spectrum.plot`, :meth:`PowerSpectrum.plot` and
    :meth:`Spectrogram.plot`. ``values`` may be 1-D (magnitude per bin) or 2-D
    (frequency x time); in both cases the boolean mask selects rows along the
    frequency axis. Returns the inputs unchanged when ``max_freq`` is ``None``.
    """
    if max_freq is None:
        return frequencies, values
    mask = frequencies <= max_freq
    return frequencies[mask], values[mask]


def top_peaks(
    frequencies: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    n: int,
    min_prominence: float | None,
) -> list[tuple[float, float]]:
    """Return up to ``n`` true local maxima as (frequency, value) pairs.

    Peaks come from :func:`scipy.signal.find_peaks`, ordered by descending
    value. Edge bins cannot qualify, so the list may be shorter than ``n``.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}.")
    peak_idx, _ = scipy_signal.find_peaks(values, prominence=min_prominence)
    order = np.argsort(values[peak_idx])[::-1][:n]
    top = peak_idx[order]
    return [(float(frequencies[i]), float(values[i])) for i in top]


def in_band_mask(
    frequencies: npt.NDArray[np.float64],
    low: float,
    high: float,
    subject: str,
) -> npt.NDArray[np.bool_]:
    """Return the boolean mask selecting ``[low, high]`` Hz from ``frequencies``.

    ``subject`` names the caller in the error message ("spectrum", "estimate"),
    so the two callers keep the wording they had before.
    """
    if low > high:
        raise ValueError(f"low ({low}) must not exceed high ({high}).")
    mask = (frequencies >= low) & (frequencies <= high)
    if not mask.any():
        spacing = frequencies[1] - frequencies[0] if len(frequencies) > 1 else 0
        raise ValueError(
            f"No frequency bins in [{low}, {high}] Hz. This {subject} spans "
            f"{frequencies[0]:g}-{frequencies[-1]:g} Hz with {spacing:g} Hz spacing."
        )
    return mask
