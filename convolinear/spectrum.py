"""Spectrum class for frequency-domain signal representations."""

from __future__ import annotations
from typing import Optional
import numpy as np


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

    def in_range(self, low: float, high: float) -> "Spectrum":
        """Return a new Spectrum containing only frequencies in [low, high] Hz."""
        mask = (self.frequencies >= low) & (self.frequencies <= high)
        return Spectrum(self.magnitudes[mask], self.frequencies[mask])

    def to_signal(self, sample_rate: int) -> "Signal":
        """Reconstruct a time-domain signal via inverse FFT.

        Because this Spectrum stores only magnitudes (no phase information),
        the reconstructed signal has zero phase - all components are cosines.
        This is useful for synthesis and spectral shaping, but is **not** a
        lossless round-trip from the original signal.

        Args:
            sample_rate: Sample rate of the output Signal in Hz.

        Returns:
            A Signal whose frequency content matches these magnitudes.
        """
        from .signal import Signal

        n_full = (len(self.magnitudes) - 1) * 2
        # Undo the 2/n normalisation applied in Signal.fft()
        coeffs = (self.magnitudes * n_full / 2.0).astype(complex)
        # DC and Nyquist bins are not doubled in rfft, so halve them back
        coeffs[0] /= 2.0
        if n_full % 2 == 0:
            coeffs[-1] /= 2.0
        data = np.fft.irfft(coeffs, n=n_full)
        return Signal(data, sample_rate)

    def plot(
        self,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        log_scale: bool = False,
        max_freq: Optional[float] = None,
        ax=None,
    ):
        """Plot the magnitude spectrum. Returns the matplotlib axis."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 3))

        freqs = self.frequencies
        mags = self.magnitudes
        if max_freq is not None:
            mask = freqs <= max_freq
            freqs = freqs[mask]
            mags = mags[mask]

        ax.plot(freqs, mags, linewidth=0.8)
        ax.set_xlabel(xlabel or "Frequency (Hz)")
        ax.set_ylabel(ylabel or "Magnitude")
        ax.set_title(title or "Frequency Spectrum")
        if log_scale:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        return ax