"""Spectrum class for frequency-domain signal representations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .signal import Signal


def _limit_to_max_freq(frequencies, values, max_freq):
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

    Produced by Signal.fft(). Holds magnitudes and their corresponding
    frequencies in Hertz.
    """

    def __init__(self, magnitudes: np.ndarray, frequencies: np.ndarray):
        self.magnitudes = np.asarray(magnitudes, dtype=np.float64)
        self.frequencies = np.asarray(frequencies, dtype=np.float64)

        if self.magnitudes.shape != self.frequencies.shape:
            raise ValueError(
                f"magnitudes and frequencies must have the same shape, "
                f"got {self.magnitudes.shape} and {self.frequencies.shape}"
            )

    def __len__(self) -> int:
        return len(self.magnitudes)

    def __repr__(self) -> str:
        return (
            f"Spectrum(bins={len(self.magnitudes)}, "
            f"freq_range=({self.frequencies[0]:.1f}, {self.frequencies[-1]:.1f}) Hz)"
        )

    @property
    def peak_frequency(self) -> float:
        """The frequency with the highest magnitude (dominant frequency)."""
        return float(self.frequencies[np.argmax(self.magnitudes)])

    @property
    def peak_magnitude(self) -> float:
        """The magnitude at the peak frequency."""
        return float(np.max(self.magnitudes))

    def top_n(self, n: int = 5) -> list[tuple[float, float]]:
        """Return the top n peaks as (frequency, magnitude) pairs."""
        idx = np.argsort(self.magnitudes)[::-1][:n]
        return [(float(self.frequencies[i]), float(self.magnitudes[i])) for i in idx]

    def in_range(self, low: float, high: float) -> Spectrum:
        """Return a new Spectrum containing only frequencies in [low, high] Hz."""
        mask = (self.frequencies >= low) & (self.frequencies <= high)
        return Spectrum(self.magnitudes[mask], self.frequencies[mask])

    def to_signal(self, sample_rate: int) -> Signal:
        """Reconstruct a time-domain signal via inverse FFT.

        Because this Spectrum stores only magnitudes (no phase information),
        the reconstructed signal has zero phase - all components are cosines.
        This is useful for synthesis and spectral shaping, but is **not** a
        lossless round-trip from the original signal.

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
        ax=None,
    ):
        """Plot the magnitude spectrum. Returns the matplotlib axis."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 3))

        freqs, mags = _limit_to_max_freq(self.frequencies, self.magnitudes, max_freq)

        ax.plot(freqs, mags, linewidth=0.8)
        ax.set_xlabel(xlabel or "Frequency (Hz)")
        ax.set_ylabel(ylabel or "Magnitude")
        ax.set_title(title or "Frequency Spectrum")
        if log_scale:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        return ax
