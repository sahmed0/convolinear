"""Tests for the PowerSpectrum class and Signal.psd (Welch's method)."""

import numpy as np
import pytest
from matplotlib.axes import Axes

from convolinear import PowerSpectrum, Signal


class TestPSD:
    def test_returns_power_spectrum(self):
        sig = Signal.noise(duration=2.0, sample_rate=1000, seed=0)
        assert isinstance(sig.psd(), PowerSpectrum)

    def test_parseval_total_power(self):
        """Integrating the PSD over frequency recovers the signal's mean power.

        Welch with ``scaling="density"`` integrates to the signal variance;
        for zero-mean unit-variance noise that matches ``Signal.power`` (mean
        square). Loose tolerance - the PSD is a statistical estimate.
        """
        sig = Signal.noise(duration=10, sample_rate=1000, seed=0)
        ps = sig.psd()
        total = np.trapezoid(ps.power, ps.frequencies)
        assert total == pytest.approx(sig.power, rel=0.1)

    def test_peak_frequency_finds_tone_in_noise(self):
        tone = Signal.sine(120, duration=10, sample_rate=1000)
        noisy = tone + Signal.noise(duration=10, sample_rate=1000, amplitude=0.3, seed=1)
        assert noisy.psd().peak_frequency == pytest.approx(120, abs=5)

    def test_rejects_unknown_window(self):
        sig = Signal.noise(duration=1.0, sample_rate=1000, seed=0)
        with pytest.raises(ValueError, match="Unknown window"):
            sig.psd(window="triangular")

    def test_rejects_overlap_out_of_range(self):
        sig = Signal.noise(duration=1.0, sample_rate=1000, seed=0)
        with pytest.raises(ValueError, match="overlap"):
            sig.psd(overlap=1.0)

    def test_rejects_non_positive_segment_length(self):
        sig = Signal.noise(duration=1.0, sample_rate=1000, seed=0)
        with pytest.raises(ValueError, match="segment_length"):
            sig.psd(segment_length=0)

    def test_segment_longer_than_signal_is_capped(self):
        sig = Signal.noise(duration=0.05, sample_rate=1000, seed=0)  # 50 samples
        ps = sig.psd(segment_length=256)
        assert len(ps) == 50 // 2 + 1


class TestPowerSpectrumConstruction:
    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="same shape"):
            PowerSpectrum(np.arange(4.0), np.arange(3.0))

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="at least one bin"):
            PowerSpectrum(np.array([]), np.array([]))

    def test_frequencies_and_power_are_read_only(self):
        ps = PowerSpectrum(np.arange(3.0), np.arange(3.0))
        with pytest.raises(ValueError):
            ps.frequencies[0] = 1.0
        with pytest.raises(ValueError):
            ps.power[0] = 1.0

    def test_does_not_freeze_callers_array(self):
        freqs = np.arange(3.0)
        power = np.ones(3)
        PowerSpectrum(freqs, power)
        power[0] = 5.0  # caller still owns its array
        assert power[0] == 5.0

    def test_len_and_repr(self):
        ps = PowerSpectrum(np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0]))
        assert len(ps) == 3
        assert "PowerSpectrum(bins=3" in repr(ps)


class TestPeaks:
    def test_peak_power(self):
        ps = PowerSpectrum(np.array([0.0, 1.0, 2.0]), np.array([1.0, 9.0, 4.0]))
        assert ps.peak_power == 9.0
        assert ps.peak_frequency == 1.0

    def test_top_n_returns_true_peaks(self):
        sig = Signal.sine(100, 5.0, 1000) + Signal.sine(250, 5.0, 1000)
        peaks = sig.psd().top_n(2)
        freqs = sorted(f for f, _ in peaks)
        assert freqs[0] == pytest.approx(100, abs=5)
        assert freqs[1] == pytest.approx(250, abs=5)

    def test_top_n_rejects_n_below_one(self):
        ps = PowerSpectrum(np.arange(3.0), np.arange(3.0))
        with pytest.raises(ValueError, match="at least 1"):
            ps.top_n(0)


class TestInRange:
    def test_subsets_to_band(self):
        freqs = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        power = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        band = PowerSpectrum(freqs, power).in_range(10, 30)
        np.testing.assert_array_equal(band.frequencies, [10.0, 20.0, 30.0])
        np.testing.assert_array_equal(band.power, [2.0, 3.0, 4.0])

    def test_rejects_low_above_high(self):
        ps = PowerSpectrum(np.arange(5.0), np.arange(5.0))
        with pytest.raises(ValueError, match="must not exceed"):
            ps.in_range(3, 1)

    def test_empty_band_raises(self):
        ps = PowerSpectrum(np.array([0.0, 10.0, 20.0]), np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError, match="No frequency bins"):
            ps.in_range(4, 6)


class TestPlot:
    def test_returns_axes(self):
        sig = Signal.noise(duration=2.0, sample_rate=1000, seed=0)
        assert isinstance(sig.psd().plot(), Axes)

    def test_draws_on_provided_axes(self):
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        sig = Signal.noise(duration=2.0, sample_rate=1000, seed=0)
        assert sig.psd().plot(ax=ax) is ax
