"""Spectrum class for frequency-domain signal representations."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Self, cast

import numpy as np
import numpy.typing as npt
from scipy import signal as scipy_signal

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .signal import Signal


def _limit_to_max_freq(
    frequencies: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    max_freq: float | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Restrict a frequency axis (and its values) to ``<= max_freq``.

    Shared by :meth:`Spectrum.plot` and :meth:`Spectrogram.plot`. ``values``
    may be 1-D (magnitude per bin) or 2-D (frequency x time); in both cases the
    boolean mask selects rows along the frequency axis. Returns the inputs
    unchanged when ``max_freq`` is ``None``.
    """
    if max_freq is None:
        return frequencies, values
    mask = frequencies <= max_freq
    return frequencies[mask], values[mask]


class Spectrum:
    """A frequency-domain representation of a signal.

    Produced by :meth:`Signal.fft`. Stores the raw complex rfft coefficients,
    so the transform is invertible: :meth:`to_signal` is a lossless inverse.
    Magnitudes (in the amplitude convention where a unit-amplitude sine reads
    ~1), phase, and power are derived views computed from the coefficients.

    The stored ``coefficients`` and ``frequencies`` arrays are read-only. The
    derived ``magnitudes``/``phase``/``power`` properties return a fresh,
    writable array on each access.
    """

    def __init__(
        self,
        coefficients: npt.ArrayLike,
        frequencies: npt.ArrayLike,
        n_samples: int,
        sample_rate: float,
        scale: float | None = None,
    ):
        """Create a Spectrum from raw rfft coefficients.

        Args:
            coefficients: The 1-D complex rfft coefficients (length
                          ``n_samples // 2 + 1``).
            frequencies:  The frequency axis in Hz, same shape as
                          ``coefficients``.
            n_samples:    Length of the original time-domain signal. Needed to
                          invert the transform and to identify the Nyquist bin.
            sample_rate:  Sample rate of the original signal in Hz.
            scale:        Magnitude scale factor. Defaults to ``2 / n_samples``
                          (the convention for an un-windowed FFT); pass
                          ``2 / window.sum()`` for a windowed spectrum.

        Raises:
            ValueError: if ``coefficients`` is not 1-D, its shape differs from
                ``frequencies``, ``n_samples`` is not positive, the coefficient
                count is inconsistent with ``n_samples``, or ``sample_rate`` is
                not finite and positive.
        """
        coeffs = np.asarray(coefficients, dtype=np.complex128)
        if coeffs is coefficients:
            coeffs = coeffs.copy()
        freqs = np.asarray(frequencies, dtype=np.float64)
        if freqs is frequencies:
            freqs = freqs.copy()

        if coeffs.ndim != 1:
            raise ValueError(f"coefficients must be 1-D, got shape {coeffs.shape}.")
        if coeffs.shape != freqs.shape:
            raise ValueError(
                f"coefficients and frequencies must have the same shape, "
                f"got {coeffs.shape} and {freqs.shape}"
            )
        if n_samples < 1:
            raise ValueError(f"n_samples must be positive, got {n_samples}.")
        if len(coeffs) != n_samples // 2 + 1:
            raise ValueError(
                f"Expected {n_samples // 2 + 1} rfft coefficients for "
                f"n_samples={n_samples}, got {len(coeffs)}."
            )
        if len(coeffs) == 0:
            raise ValueError("Spectrum must contain at least one bin.")
        if not math.isfinite(sample_rate) or sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive and finite, got {sample_rate}.")

        coeffs.flags.writeable = False
        freqs.flags.writeable = False
        self._coefficients = coeffs
        self._frequencies = freqs
        self._n_samples = int(n_samples)
        self._sample_rate = float(sample_rate)
        self._scale = 2.0 / n_samples if scale is None else float(scale)

    def __len__(self) -> int:
        return len(self._coefficients)

    def __repr__(self) -> str:
        return (
            f"Spectrum(bins={len(self._coefficients)}, "
            f"freq_range=({self._frequencies[0]:.1f}, {self._frequencies[-1]:.1f}) Hz)"
        )

    @property
    def coefficients(self) -> npt.NDArray[np.complex128]:
        """The raw complex rfft coefficients (read-only)."""
        return self._coefficients

    @property
    def frequencies(self) -> npt.NDArray[np.float64]:
        """The frequency axis in Hz (read-only)."""
        return self._frequencies

    @property
    def n_samples(self) -> int:
        """Length of the original time-domain signal."""
        return self._n_samples

    @property
    def sample_rate(self) -> float:
        """Sample rate of the original signal in Hz."""
        return self._sample_rate

    @property
    def magnitudes(self) -> npt.NDArray[np.float64]:
        """Single-sided amplitudes under the library's convention (unit sine ~ 1).

        Returns a fresh writable array each access.
        """
        mags: npt.NDArray[np.float64] = np.abs(self._coefficients) * self._scale
        mags[0] /= 2.0  # DC is not split across +/- pairs
        if self._n_samples % 2 == 0 and len(mags) > 1:
            mags[-1] /= 2.0  # even-N Nyquist bin likewise
        return mags

    @property
    def phase(self) -> npt.NDArray[np.float64]:
        """Phase angle of each bin in radians, in (-pi, pi]."""
        return cast("npt.NDArray[np.float64]", np.angle(self._coefficients))

    @property
    def power(self) -> npt.NDArray[np.float64]:
        """Per-bin power: ``magnitudes ** 2`` (amplitude-squared, not density -
        for a density estimate see :meth:`Signal.psd`)."""
        return self.magnitudes**2

    @property
    def peak_frequency(self) -> float:
        """The frequency with the highest magnitude (dominant frequency)."""
        return float(self._frequencies[np.argmax(self.magnitudes)])

    @property
    def peak_magnitude(self) -> float:
        """The magnitude at the peak frequency."""
        return float(np.max(self.magnitudes))

    def top_n(self, n: int = 5, min_prominence: float | None = None) -> list[tuple[float, float]]:
        """Return up to ``n`` spectral peaks as (frequency, magnitude) pairs.

        Peaks are true local maxima found with :func:`scipy.signal.find_peaks`
        (not the ``n`` largest bins - the largest bins of a leaky spectrum are
        usually samples of a single lobe). Results are ordered by descending
        magnitude. The list may contain fewer than ``n`` entries - or be empty -
        when the spectrum has fewer qualifying peaks. Bins at the very edges of
        the spectrum (DC and Nyquist) cannot qualify as peaks.

        Args:
            n:              Maximum number of peaks to return.
            min_prominence: Optional prominence threshold, in magnitude units,
                            forwarded to ``find_peaks``. Use it to suppress noise
                            peaks.

        Raises:
            ValueError: if ``n`` is less than 1.
        """
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}.")
        magnitudes = self.magnitudes
        peak_idx, _ = scipy_signal.find_peaks(magnitudes, prominence=min_prominence)
        order = np.argsort(magnitudes[peak_idx])[::-1][:n]
        top = peak_idx[order]
        return [(float(self._frequencies[i]), float(magnitudes[i])) for i in top]

    def in_range(self, low: float, high: float) -> Spectrum:
        """Zero every bin outside [low, high] Hz (an ideal brick-wall selection).

        The returned Spectrum keeps the full frequency axis and bin count, so it
        remains a complete, invertible rfft: ``in_range(...).to_signal()`` is the
        signal with all out-of-band content removed.

        Note that :meth:`top_n` on a brick-walled spectrum can report a bin at
        the very edge of the retained band as a peak (the sharp cut creates a
        local maximum); this is expected and not special-cased.

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
                f"No frequency bins in [{low}, {high}] Hz. This spectrum spans "
                f"{self._frequencies[0]:g}-{self._frequencies[-1]:g} Hz with "
                f"{spacing:g} Hz spacing."
            )
        coeffs = np.where(mask, self._coefficients, 0)
        return Spectrum(coeffs, self._frequencies, self._n_samples, self._sample_rate, self._scale)

    def to_signal(self) -> Signal:
        """Invert the FFT: reconstruct the exact time-domain signal.

        Lossless: for ``spec = sig.fft()``, ``spec.to_signal()`` equals ``sig``
        to floating-point precision. For a windowed spectrum
        (``sig.fft(window=...)``) the reconstruction is the *windowed* waveform -
        the taper is part of the transformed data and is faithfully returned; it
        is not divided back out. For zero-phase synthesis from a magnitude
        spectrum, see :meth:`from_magnitudes`.

        Returns:
            A Signal reconstructed by :func:`numpy.fft.irfft`, carrying this
            spectrum's sample rate.
        """
        from .signal import Signal

        return Signal(np.fft.irfft(self._coefficients, n=self._n_samples), self._sample_rate)


        Each magnitude is placed at its true frequency, derived from the bin
        spacing, so a sub-band Spectrum (e.g. one returned by :meth:`in_range`)
        synthesises back at the correct frequencies rather than being shifted
        down to DC. A Spectrum produced by ``Signal.fft(window=...)`` carries
        the window's coherent-gain correction, so its reconstructed amplitudes
        will be scaled by that window and are not directly recoverable.

        Args:
            sample_rate: Sample rate of the output Signal in Hz.

        Returns:
            A Signal whose frequency content matches these magnitudes.

        Raises:
            ValueError: if the Spectrum has fewer than two bins (the bin
                spacing, and hence the signal length, cannot be inferred).
        """
        from .signal import Signal

        if len(self.magnitudes) < 2:
            raise ValueError(
                "Need at least two frequency bins to reconstruct a signal; "
                f"got {len(self.magnitudes)}. A single-bin Spectrum carries no "
                "recoverable time-domain information."
            )

        # Recover the FFT bin spacing (Hz/bin). Frequencies from Signal.fft()
        # are evenly spaced, so the spacing plus the target sample rate fix the
        # full transform length - which also recovers odd original lengths.
        df = float(self.frequencies[1] - self.frequencies[0])
        if df <= 0:
            raise ValueError("Spectrum frequencies must be evenly spaced and increasing.")
        n_full = round(sample_rate / df)
        if n_full < 2:
            raise ValueError(
                f"sample_rate {sample_rate} Hz is too low for this Spectrum's "
                f"bin spacing ({df} Hz)."
            )
        n_bins = n_full // 2 + 1

        # Scatter each magnitude into its true rfft bin; bins absent from this
        # Spectrum (e.g. dropped by in_range) stay zero.
        coeffs = np.zeros(n_bins, dtype=complex)
        bin_indices = np.round(self.frequencies / df).astype(int)
        in_bounds = (bin_indices >= 0) & (bin_indices < n_bins)
        # Undo the interior-bin 2/n normalisation applied in Signal.fft().
        coeffs[bin_indices[in_bounds]] = self.magnitudes[in_bounds] * n_full / 2.0
        # fft() stores DC (and, for even n, Nyquist) un-doubled, so scale those
        # bins back up to their full rfft coefficient.
        coeffs[0] *= 2.0
        if n_full % 2 == 0:
            coeffs[-1] *= 2.0
        data = np.fft.irfft(coeffs, n=n_full)
        return Signal(data, sample_rate)

    def plot(
        self,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        log_scale: bool = False,
        max_freq: float | None = None,
        ax: Axes | None = None,
    ) -> Axes:
        """Plot the magnitude spectrum. Returns the matplotlib axis."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 3))

        freqs, mags = _limit_to_max_freq(self._frequencies, self.magnitudes, max_freq)

        ax.plot(freqs, mags, linewidth=0.8)
        ax.set_xlabel(xlabel or "Frequency (Hz)")
        ax.set_ylabel(ylabel or "Magnitude")
        ax.set_title(title or "Frequency Spectrum")
        if log_scale:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        return ax
