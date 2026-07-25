"""Tests for the Spectrum class (FFT results, to_signal, in_range, top_n, peaks)."""

import itertools

import numpy as np
import pytest

from convolinear import Signal, Spectrum


class TestFFT:
    def test_fft_returns_spectrum(self):
        sig = Signal.sine(440, duration=1.0, sample_rate=8000)
        spec = sig.fft()
        assert isinstance(spec, Spectrum)

    def test_fft_detects_sine_frequency(self):
        """The FFT of a pure 440 Hz tone should peak at 440 Hz."""
        sig = Signal.sine(440, duration=1.0, sample_rate=8000)
        spec = sig.fft()
        assert spec.peak_frequency == pytest.approx(440, abs=2)

    def test_fft_detects_two_tones(self):
        """A mix of 200 Hz and 800 Hz should show two peaks."""
        t = np.arange(8000) / 8000
        mixed = np.sin(2 * np.pi * 200 * t) + np.sin(2 * np.pi * 800 * t)
        spec = Signal(mixed, sample_rate=8000).fft()
        top = spec.top_n(2)
        peak_freqs = sorted([f for f, _ in top])
        assert peak_freqs[0] == pytest.approx(200, abs=2)
        assert peak_freqs[1] == pytest.approx(800, abs=2)


class TestSpectrum:
    def test_in_range(self):
        sig = Signal.sine(440, duration=1.0, sample_rate=8000)
        spec = sig.fft()
        sub = spec.in_range(100, 1000)
        assert sub.frequencies.min() >= 100
        assert sub.frequencies.max() <= 1000


class TestTopN:
    def test_returns_distinct_lobes(self):
        # An off-bin tone leaks across several bins; the top 3 raw bins would be
        # samples of one lobe. Real peak-picking must return distinct peaks.
        spec = Signal.sine(440.5, duration=1.0, sample_rate=8000).fft()
        peaks = spec.top_n(3)
        bin_spacing = float(spec.frequencies[1] - spec.frequencies[0])
        freqs = sorted(f for f, _ in peaks)
        for a, b in itertools.pairwise(freqs):
            assert b - a > 2 * bin_spacing

    def test_two_tone_returns_both(self):
        base = Signal.sine(440, duration=1.0, sample_rate=8000)
        second = Signal.sine(1000, duration=1.0, sample_rate=8000, amplitude=0.5)
        spec = (base + second).fft()
        peaks = spec.top_n(3)
        top_two = sorted(f for f, _ in peaks[:2])
        assert top_two[0] == pytest.approx(440, abs=2)
        assert top_two[1] == pytest.approx(1000, abs=2)

    def test_rejects_n_below_one(self):
        spec = Signal.sine(440, duration=1.0, sample_rate=8000).fft()
        with pytest.raises(ValueError, match="at least 1"):
            spec.top_n(0)



class TestFFTWindowing:
    def test_no_window_matches_default(self):
        sig = Signal.sine(440, duration=1.0, sample_rate=8000)
        spec_none = sig.fft(window=None)
        spec_default = sig.fft()
        np.testing.assert_array_equal(spec_none.magnitudes, spec_default.magnitudes)

    def test_hann_peak_frequency_preserved(self):
        sig = Signal.sine(440, duration=1.0, sample_rate=8000)
        spec = sig.fft(window="hann")
        assert spec.peak_frequency == pytest.approx(440, abs=2)

    def test_unknown_window_raises(self):
        sig = Signal.sine(440, duration=0.1, sample_rate=8000)
        with pytest.raises(ValueError, match="Unknown window"):
            sig.fft(window="kaiser")

    @pytest.mark.parametrize("win", ["hann", "hamming", "blackman", "bartlett"])
    def test_all_windows_run_without_error(self, win):
        sig = Signal.sine(440, duration=0.1, sample_rate=8000)
        spec = sig.fft(window=win)
        assert isinstance(spec, Spectrum)

    def test_window_case_insensitive(self):
        sig = Signal.sine(440, duration=0.1, sample_rate=8000)
        spec = sig.fft(window="HANN")
        assert isinstance(spec, Spectrum)


class TestSpectrumToSignal:
    def test_returns_signal_instance(self):
        spec = Signal.sine(440, duration=1.0, sample_rate=8000).fft()
        assert isinstance(spec.to_signal(8000), Signal)

    def test_sample_rate_propagated(self):
        spec = Signal.sine(440, duration=1.0, sample_rate=8000).fft()
        out = spec.to_signal(8000)
        assert out.sample_rate == 8000

    def test_output_length(self):
        sig = Signal.sine(440, duration=1.0, sample_rate=8000)
        out = sig.fft().to_signal(8000)
        assert len(out) == len(sig)

    def test_dominant_frequency_preserved(self):
        sig = Signal.sine(440, duration=1.0, sample_rate=8000)
        reconstructed = sig.fft().to_signal(8000)
        assert reconstructed.fft().peak_frequency == pytest.approx(440, abs=2)
