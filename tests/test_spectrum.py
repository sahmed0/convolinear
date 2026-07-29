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

        assert len(sub) == len(spec)
        np.testing.assert_array_equal(sub.frequencies, spec.frequencies)
        out_of_band = (sub.frequencies < 100) | (sub.frequencies > 1000)
        assert np.all(sub.magnitudes[out_of_band] == 0.0)
        in_band = ~out_of_band
        assert np.any(sub.magnitudes[in_band] > 0.0)

    def test_in_range_rejects_low_above_high(self):
        spec = Signal.sine(440, duration=1.0, sample_rate=8000).fft()
        with pytest.raises(ValueError, match="must not exceed"):
            spec.in_range(1000, 100)

    def test_in_range_empty_band_raises(self):
        # 8000 samples at 8000 Hz -> 1 Hz bin spacing; nothing sits between
        # two adjacent integer bins.
        spec = Signal.sine(440, duration=1.0, sample_rate=8000).fft()
        with pytest.raises(ValueError, match="No frequency bins"):
            spec.in_range(100.2, 100.8)


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


class TestSpectrumConstruction:
    def test_rejects_non_positive_n_samples(self):
        with pytest.raises(ValueError, match="n_samples must be positive"):
            Spectrum(np.array([1.0 + 0j]), np.array([0.0]), n_samples=0, sample_rate=8000.0)

    def test_rejects_inconsistent_coefficient_count(self):
        # 8 samples -> 5 rfft bins expected; supplying 3 must raise.
        with pytest.raises(ValueError, match="rfft coefficients"):
            Spectrum(
                np.zeros(3, dtype=complex),
                np.zeros(3),
                n_samples=8,
                sample_rate=8000.0,
            )

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="same shape"):
            Spectrum(
                np.zeros(3, dtype=complex),
                np.zeros(4),
                n_samples=4,
                sample_rate=8000.0,
            )


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
        assert isinstance(spec.to_signal(), Signal)

    def test_sample_rate_propagated(self):
        spec = Signal.sine(440, duration=1.0, sample_rate=8000).fft()
        out = spec.to_signal()
        assert out.sample_rate == 8000

    def test_output_length(self):
        sig = Signal.sine(440, duration=1.0, sample_rate=8000)
        out = sig.fft().to_signal()
        assert len(out) == len(sig)

    def test_dominant_frequency_preserved(self):
        sig = Signal.sine(440, duration=1.0, sample_rate=8000)
        reconstructed = sig.fft().to_signal()
        assert reconstructed.fft().peak_frequency == pytest.approx(440, abs=2)


class TestLosslessRoundTrip:
    @pytest.mark.parametrize("n", [64, 65])  # even and odd lengths
    def test_fft_roundtrip_is_lossless(self, n):
        rng = np.random.default_rng(0)
        sig = Signal(rng.standard_normal(n), sample_rate=8000.0)
        rt = sig.fft().to_signal()
        np.testing.assert_allclose(rt.data, sig.data, atol=1e-12)
        assert rt.sample_rate == sig.sample_rate
        assert len(rt) == len(sig)

    def test_windowed_roundtrip_returns_windowed_waveform(self):
        rng = np.random.default_rng(1)
        sig = Signal(rng.standard_normal(128), sample_rate=8000.0)
        rt = sig.fft(window="hann").to_signal()
        np.testing.assert_allclose(rt.data, sig.window("hann").data, atol=1e-12)


class TestDerivedProperties:
    def test_magnitudes_unit_sine_reads_one(self):
        # A tone at an exact bin reads magnitude ~1 under the library convention.
        spec = Signal.sine(500, duration=1.0, sample_rate=8000).fft()
        assert spec.peak_frequency == pytest.approx(500, abs=1e-6)
        assert spec.peak_magnitude == pytest.approx(1.0, abs=1e-3)

    def test_magnitudes_dc_only_signal(self):
        spec = Signal(np.ones(64), sample_rate=8000.0).fft()
        assert spec.frequencies[0] == 0.0
        assert spec.magnitudes[0] == pytest.approx(1.0)
        # No spurious energy anywhere else.
        assert np.all(spec.magnitudes[1:] < 1e-9)

    def test_magnitudes_are_a_fresh_writable_copy(self):
        spec = Signal.sine(500, duration=0.1, sample_rate=8000).fft()
        mags = spec.magnitudes
        mags[0] = 999.0  # must not raise and must not affect the spectrum
        assert spec.magnitudes[0] != 999.0

    def test_phase_of_cosine_is_zero_at_its_bin(self):
        # A cosine at an exact bin has ~zero phase there.
        spec = Signal.from_function(
            lambda t: np.cos(2 * np.pi * 500 * t), duration=1.0, sample_rate=8000
        ).fft()
        bin_idx = int(np.argmin(np.abs(spec.frequencies - 500)))
        assert spec.phase[bin_idx] == pytest.approx(0.0, abs=1e-3)

    def test_power_is_magnitudes_squared(self):
        spec = Signal.sine(500, duration=0.1, sample_rate=8000).fft()
        np.testing.assert_allclose(spec.power, spec.magnitudes**2)


class TestInRangeBrickWall:
    def test_acts_as_bandpass_filter(self):
        t = np.arange(8000) / 8000
        two_tone = np.sin(2 * np.pi * 200 * t) + np.sin(2 * np.pi * 1500 * t)
        spec = Signal(two_tone, sample_rate=8000).fft()
        filtered = spec.in_range(100, 400).to_signal()
        # Only the 200 Hz tone survives.
        assert filtered.fft().peak_frequency == pytest.approx(200, abs=2)
        # The 1500 Hz bin is gone.
        out_spec = filtered.fft()
        bin_1500 = int(np.argmin(np.abs(out_spec.frequencies - 1500)))
        assert out_spec.magnitudes[bin_1500] < 1e-6

    def test_keeps_full_length_and_zeroes_out_of_band(self):
        spec = Signal.sine(440, duration=1.0, sample_rate=8000).fft()
        sub = spec.in_range(100, 1000)
        assert len(sub) == len(spec)
        out_of_band = (sub.frequencies < 100) | (sub.frequencies > 1000)
        assert np.all(sub.magnitudes[out_of_band] == 0.0)


class TestFromMagnitudes:
    def test_reproduces_v01_cosine_synthesis(self):
        # A single magnitude of 1.0 at 500 Hz should synthesise a unit-amplitude
        # cosine at 500 Hz.
        sample_rate = 8000.0
        freqs = np.fft.rfftfreq(8000, d=1.0 / sample_rate)
        mags = np.zeros_like(freqs)
        mags[np.argmin(np.abs(freqs - 500))] = 1.0
        sig = Spectrum.from_magnitudes(mags, freqs, sample_rate).to_signal()
        assert float(np.max(np.abs(sig.data))) == pytest.approx(1.0, abs=1e-6)
        assert sig.fft().peak_frequency == pytest.approx(500, abs=1)

    def test_full_axis_from_subband_input(self):
        sample_rate = 8000.0
        freqs = np.fft.rfftfreq(8000, d=1.0 / sample_rate)
        # Take only a sub-band as input; the synthesised spectrum spans the full
        # rfft grid and places the tone at its true frequency.
        band = (freqs >= 400) & (freqs <= 600)
        mags = np.zeros(band.sum())
        sub_freqs = freqs[band]
        mags[np.argmin(np.abs(sub_freqs - 500))] = 1.0
        spec = Spectrum.from_magnitudes(mags, sub_freqs, sample_rate)
        assert spec.frequencies[0] == 0.0
        assert spec.to_signal().fft().peak_frequency == pytest.approx(500, abs=1)

    def test_rejects_single_bin(self):
        with pytest.raises(ValueError, match="at least two"):
            Spectrum.from_magnitudes(np.array([1.0]), np.array([0.0]), 8000.0)


class TestImmutability:
    def test_coefficients_are_read_only(self):
        spec = Signal.sine(500, duration=0.1, sample_rate=8000).fft()
        with pytest.raises(ValueError):
            spec.coefficients[0] = 1.0

    def test_frequencies_are_read_only(self):
        spec = Signal.sine(500, duration=0.1, sample_rate=8000).fft()
        with pytest.raises(ValueError):
            spec.frequencies[0] = 1.0

    def test_cannot_reassign_coefficients(self):
        spec = Signal.sine(500, duration=0.1, sample_rate=8000).fft()
        with pytest.raises(AttributeError):
            spec.coefficients = np.zeros(3, dtype=complex)
