"""Tests for the Signal class: construction, properties, transformations,
operators, filters, convolution/correlation/delay, peaks, and generators."""

import numpy as np
import pytest

from convolinear import Signal


class TestSignalConstruction:
    def test_basic_construction(self):
        data = np.array([0.0, 0.5, -0.5, 1.0])
        sig = Signal(data, sample_rate=4)
        assert len(sig) == 4
        assert sig.sample_rate == 4
        assert sig.duration == 1.0

    def test_rejects_2d_data(self):
        with pytest.raises(ValueError, match="must be 1D"):
            Signal(np.array([[1.0, 2.0], [3.0, 4.0]]), sample_rate=2)

    def test_rejects_negative_sample_rate(self):
        with pytest.raises(ValueError, match="positive"):
            Signal(np.array([1.0, 2.0]), sample_rate=-1)

    def test_sine_constructor(self):
        sig = Signal.sine(frequency=440, duration=0.5, sample_rate=8000)
        assert len(sig) == 4000
        assert sig.sample_rate == 8000

    def test_from_function(self):
        sig = Signal.from_function(lambda t: t * 2, duration=1.0, sample_rate=10)
        assert len(sig) == 10
        # First sample at t=0 should be 0
        assert sig.data[0] == pytest.approx(0.0)


class TestSignalTransformations:
    def test_normalize(self):
        sig = Signal(np.array([0.0, 0.5, -0.25, 0.3]), sample_rate=4)
        normalized = sig.normalize()
        assert np.max(np.abs(normalized.data)) == pytest.approx(1.0)

    def test_normalize_handles_zero_signal(self):
        sig = Signal(np.zeros(10), sample_rate=10)
        normalized = sig.normalize()
        assert np.all(normalized.data == 0)

    def test_trim(self):
        sig = Signal.sine(440, duration=1.0, sample_rate=1000)
        trimmed = sig.trim(start=0.2, end=0.6)
        assert len(trimmed) == 400
        assert trimmed.duration == pytest.approx(0.4)

    def test_gain(self):
        sig = Signal(np.array([1.0, 2.0, 3.0]), sample_rate=3)
        amplified = sig.gain(2.0)
        np.testing.assert_array_almost_equal(amplified.data, [2.0, 4.0, 6.0])

    def test_gain_db(self):
        sig = Signal(np.array([1.0]), sample_rate=1)
        # +6dB ≈ 2x amplitude
        assert sig.gain_db(6).data[0] == pytest.approx(2.0, rel=0.01)

    def test_immutability(self):
        """Transformations must return new Signals, not mutate the original."""
        original = Signal(np.array([1.0, 2.0, 3.0]), sample_rate=3)
        original_data = original.data.copy()
        _ = original.normalize().gain(2.0)
        np.testing.assert_array_equal(original.data, original_data)

    def test_chaining(self):
        """Verify the fluent API actually works."""
        result = (
            Signal.sine(1000, duration=0.5, sample_rate=8000).gain(0.5).normalize().trim(0.1, 0.4)
        )
        assert isinstance(result, Signal)
        assert result.duration == pytest.approx(0.3)


class TestFiltering:
    def test_lowpass_removes_high_frequency(self):
        """A low-pass filter should attenuate high frequencies."""
        # Mix a 100 Hz tone and a 3000 Hz tone
        t = np.arange(8000) / 8000
        mixed = np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 3000 * t)
        sig = Signal(mixed, sample_rate=8000)

        filtered = sig.lowpass(cutoff=500)
        # After low-pass at 500 Hz, the 3000 Hz component should be much smaller
        original_spectrum = sig.fft()
        filtered_spectrum = filtered.fft()

        # Find magnitudes near 3000 Hz
        orig_high = original_spectrum.in_range(2900, 3100).peak_magnitude
        filt_high = filtered_spectrum.in_range(2900, 3100).peak_magnitude
        assert filt_high < orig_high * 0.1  # at least 10x attenuation

    def test_cutoff_out_of_range_raises(self):
        sig = Signal.sine(440, duration=0.1, sample_rate=8000)
        with pytest.raises(ValueError, match="out of range"):
            sig.lowpass(cutoff=5000)  # above Nyquist (4000 Hz)

    def test_bandpass(self):
        sig = Signal.sine(440, duration=0.5, sample_rate=8000)
        filtered = sig.bandpass(low=300, high=600)
        assert isinstance(filtered, Signal)
        # Energy should be preserved roughly since 440 Hz is in the band
        assert np.std(filtered.data) > 0.5 * np.std(sig.data)


class TestHighpass:
    def test_highpass_removes_low_frequency(self):
        t = np.arange(8000) / 8000
        mixed = np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 3000 * t)
        sig = Signal(mixed, sample_rate=8000)
        filtered = sig.highpass(cutoff=500)
        orig_low = sig.fft().in_range(50, 150).peak_magnitude
        filt_low = filtered.fft().in_range(50, 150).peak_magnitude
        assert filt_low < orig_low * 0.1  # at least 10x attenuation

    def test_cutoff_out_of_range_raises(self):
        sig = Signal.sine(440, duration=0.1, sample_rate=8000)
        with pytest.raises(ValueError, match="out of range"):
            sig.highpass(cutoff=5000)  # above Nyquist (4000 Hz)


class TestResample:
    def test_upsample_length(self):
        sig = Signal.sine(440, duration=1.0, sample_rate=8000)
        up = sig.resample(16000)
        assert len(up) == 16000

    def test_downsample_length(self):
        sig = Signal.sine(440, duration=1.0, sample_rate=8000)
        down = sig.resample(4000)
        assert len(down) == 4000

    def test_new_sample_rate_set(self):
        sig = Signal.sine(440, duration=0.5, sample_rate=8000)
        resampled = sig.resample(22050)
        assert resampled.sample_rate == 22050


class TestNoise:
    def test_length_and_sample_rate(self):
        sig = Signal.noise(duration=0.5, sample_rate=1000)
        assert len(sig) == 500
        assert sig.sample_rate == 1000

    def test_seed_is_reproducible(self):
        sig1 = Signal.noise(duration=0.1, sample_rate=1000, seed=42)
        sig2 = Signal.noise(duration=0.1, sample_rate=1000, seed=42)
        np.testing.assert_array_equal(sig1.data, sig2.data)

    def test_amplitude_scales_output(self):
        sig1 = Signal.noise(duration=1.0, sample_rate=8000, amplitude=1.0, seed=0)
        sig2 = Signal.noise(duration=1.0, sample_rate=8000, amplitude=2.0, seed=0)
        assert np.std(sig2.data) == pytest.approx(2.0 * np.std(sig1.data), rel=1e-9)


class TestTimeAxis:
    def test_length_matches_signal(self):
        sig = Signal.sine(440, duration=0.5, sample_rate=8000)
        assert len(sig.time_axis) == len(sig)

    def test_starts_at_zero(self):
        sig = Signal(np.array([1.0, 2.0, 3.0]), sample_rate=10)
        assert sig.time_axis[0] == 0.0

    def test_last_value(self):
        n, sr = 100, 1000
        sig = Signal(np.zeros(n), sample_rate=sr)
        assert sig.time_axis[-1] == pytest.approx((n - 1) / sr)


# ─── v0.2.0 feature tests ─────────────────────────────────────────────────────


class TestMixing:
    """Tests for Signal.__add__ (mixing operator)."""

    def test_add_element_wise(self):
        a = Signal(np.array([1.0, 2.0, 3.0]), sample_rate=3)
        b = Signal(np.array([0.1, 0.2, 0.3]), sample_rate=3)
        mixed = a + b
        np.testing.assert_array_almost_equal(mixed.data, [1.1, 2.2, 3.3])

    def test_add_same_sample_rate_preserved(self):
        a = Signal.sine(440, duration=0.1, sample_rate=8000)
        b = Signal.sine(880, duration=0.1, sample_rate=8000)
        assert (a + b).sample_rate == 8000

    def test_add_pads_shorter_signal(self):
        long_sig = Signal(np.ones(10), sample_rate=10)
        short_sig = Signal(np.ones(4), sample_rate=10)
        result = long_sig + short_sig
        assert len(result) == 10
        np.testing.assert_array_almost_equal(result.data[:4], 2.0)
        np.testing.assert_array_almost_equal(result.data[4:], 1.0)

    def test_add_commutativity(self):
        a = Signal(np.array([1.0, 2.0, 3.0]), sample_rate=3)
        b = Signal(np.array([0.5, 0.5, 0.5]), sample_rate=3)
        np.testing.assert_array_almost_equal((a + b).data, (b + a).data)

    def test_add_mismatched_sample_rate_raises(self):
        a = Signal.sine(440, duration=0.1, sample_rate=8000)
        b = Signal.sine(440, duration=0.1, sample_rate=44100)
        with pytest.raises(ValueError, match="sample rate"):
            _ = a + b

    def test_add_non_signal_returns_not_implemented(self):
        sig = Signal(np.array([1.0, 2.0]), sample_rate=2)
        assert sig.__add__(42) is NotImplemented

    def test_add_returns_signal_instance(self):
        a = Signal(np.array([1.0]), sample_rate=1)
        b = Signal(np.array([1.0]), sample_rate=1)
        assert isinstance(a + b, Signal)


class TestConcat:
    """Tests for Signal.concat()."""

    def test_concat_length(self):
        a = Signal(np.ones(4), sample_rate=4)
        b = Signal(np.ones(6), sample_rate=4)
        assert len(a.concat(b)) == 10

    def test_concat_data_order(self):
        a = Signal(np.array([1.0, 2.0]), sample_rate=2)
        b = Signal(np.array([3.0, 4.0]), sample_rate=2)
        np.testing.assert_array_almost_equal(a.concat(b).data, [1.0, 2.0, 3.0, 4.0])

    def test_concat_sample_rate_preserved(self):
        a = Signal(np.ones(4), sample_rate=8000)
        b = Signal(np.ones(4), sample_rate=8000)
        assert a.concat(b).sample_rate == 8000

    def test_concat_mismatched_rate_raises(self):
        a = Signal(np.ones(4), sample_rate=8000)
        b = Signal(np.ones(4), sample_rate=44100)
        with pytest.raises(ValueError, match="sample rate"):
            a.concat(b)

    def test_concat_non_signal_raises(self):
        sig = Signal(np.ones(4), sample_rate=4)
        with pytest.raises(TypeError):
            sig.concat([1.0, 2.0])

    def test_concat_is_not_commutative(self):
        a = Signal(np.array([1.0, 2.0]), sample_rate=2)
        b = Signal(np.array([3.0, 4.0]), sample_rate=2)
        assert not np.array_equal(a.concat(b).data, b.concat(a).data)


class TestAmplitudeProperties:
    """Tests for Signal.rms, Signal.rms_db, Signal.power, Signal.power_db,
    and Signal.peak_db."""

    def test_rms_full_scale_sine(self):
        sig = Signal.sine(440, duration=1.0, sample_rate=44100, amplitude=1.0)
        assert sig.rms == pytest.approx(1.0 / np.sqrt(2), rel=1e-3)

    def test_rms_constant_signal(self):
        sig = Signal(np.full(100, 2.0), sample_rate=100)
        assert sig.rms == pytest.approx(2.0)

    def test_rms_zero_signal(self):
        sig = Signal(np.zeros(100), sample_rate=100)
        assert sig.rms == pytest.approx(0.0)

    def test_rms_db_full_scale_sine(self):
        sig = Signal.sine(440, duration=1.0, sample_rate=44100, amplitude=1.0)
        # RMS of a sine = 1/√2 → 20*log10(1/√2) ≈ -3.0103 dB
        assert sig.rms_db == pytest.approx(-3.0103, abs=0.05)

    def test_rms_db_zero_signal(self):
        sig = Signal(np.zeros(100), sample_rate=100)
        assert sig.rms_db == float("-inf")

    def test_power_full_scale_sine(self):
        sig = Signal.sine(440, duration=1.0, sample_rate=44100, amplitude=1.0)
        # Power of a sine = (1/√2)² = 0.5
        assert sig.power == pytest.approx(0.5, rel=1e-3)

    def test_power_constant_signal(self):
        sig = Signal(np.full(100, 2.0), sample_rate=100)
        assert sig.power == pytest.approx(4.0)

    def test_power_zero_signal(self):
        sig = Signal(np.zeros(100), sample_rate=100)
        assert sig.power == pytest.approx(0.0)

    def test_power_is_rms_squared(self):
        sig = Signal.sine(440, duration=1.0, sample_rate=44100, amplitude=0.7)
        assert sig.power == pytest.approx(sig.rms**2)

    def test_power_db_full_scale_sine(self):
        sig = Signal.sine(440, duration=1.0, sample_rate=44100, amplitude=1.0)
        # Power = 0.5 → 10*log10(0.5) ≈ -3.0103 dB
        assert sig.power_db == pytest.approx(-3.0103, abs=0.05)

    def test_power_db_equals_rms_db(self):
        sig = Signal.sine(440, duration=1.0, sample_rate=44100, amplitude=0.7)
        assert sig.power_db == pytest.approx(sig.rms_db)

    def test_power_db_zero_signal(self):
        sig = Signal(np.zeros(100), sample_rate=100)
        assert sig.power_db == float("-inf")

    def test_peak_db_unit_amplitude(self):
        sig = Signal(np.array([1.0, -0.5, 0.2]), sample_rate=3)
        assert sig.peak_db == pytest.approx(0.0)

    def test_peak_db_half_amplitude(self):
        sig = Signal(np.array([0.5, -0.5]), sample_rate=2)
        assert sig.peak_db == pytest.approx(20 * np.log10(0.5))

    def test_peak_db_zero_signal(self):
        sig = Signal(np.zeros(10), sample_rate=10)
        assert sig.peak_db == float("-inf")


class TestRemoveDC:
    def test_remove_dc_zeroes_mean(self):
        sig = Signal(np.array([1.1, 1.5, 0.9, 1.3]), sample_rate=4)
        assert sig.remove_dc().data.mean() == pytest.approx(0.0, abs=1e-12)

    def test_remove_dc_preserves_length_and_rate(self):
        sig = Signal(np.array([1.0, 2.0, 3.0]), sample_rate=10)
        out = sig.remove_dc()
        assert len(out) == 3
        assert out.sample_rate == 10

    def test_remove_dc_zero_input_unchanged(self):
        sig = Signal(np.zeros(5), sample_rate=5)
        np.testing.assert_array_almost_equal(sig.remove_dc().data, np.zeros(5))

    def test_remove_dc_returns_new_signal(self):
        sig = Signal(np.array([1.0, 2.0, 3.0]), sample_rate=3)
        assert sig.remove_dc() is not sig


class TestReverse:
    def test_reverse_known_array(self):
        sig = Signal(np.array([1.0, 2.0, 3.0, 4.0]), sample_rate=4)
        np.testing.assert_array_equal(sig.reverse().data, [4.0, 3.0, 2.0, 1.0])

    def test_reverse_twice_is_identity(self):
        sig = Signal.sine(440, duration=0.1, sample_rate=8000)
        np.testing.assert_array_almost_equal(sig.reverse().reverse().data, sig.data)

    def test_reverse_preserves_sample_rate(self):
        sig = Signal(np.array([1.0, 2.0, 3.0]), sample_rate=42)
        assert sig.reverse().sample_rate == 42


class TestClip:
    def test_clip_clamps_above_max(self):
        sig = Signal(np.array([1.5, -0.5, 0.8]), sample_rate=3)
        out = sig.clip(-1.0, 1.0)
        assert out.data[0] == pytest.approx(1.0)

    def test_clip_clamps_below_min(self):
        sig = Signal(np.array([-1.5, 0.5, 0.3]), sample_rate=3)
        out = sig.clip(-1.0, 1.0)
        assert out.data[0] == pytest.approx(-1.0)

    def test_clip_in_range_unchanged(self):
        sig = Signal(np.array([0.3, -0.3, 0.5]), sample_rate=3)
        np.testing.assert_array_almost_equal(sig.clip(-1.0, 1.0).data, sig.data)

    def test_clip_custom_bounds(self):
        sig = Signal(np.array([0.8, 0.1, -0.2]), sample_rate=3)
        out = sig.clip(0.0, 0.5)
        assert out.data[0] == pytest.approx(0.5)
        assert out.data[2] == pytest.approx(0.0)

    def test_clip_invalid_bounds_raises(self):
        sig = Signal(np.array([1.0, 2.0]), sample_rate=2)
        with pytest.raises(ValueError):
            sig.clip(1.0, 0.5)


class TestFades:
    def test_fade_in_first_sample_is_zero(self):
        sig = Signal(np.ones(1000), sample_rate=1000)
        faded = sig.fade_in(0.5)
        assert faded.data[0] == pytest.approx(0.0)

    def test_fade_in_end_of_ramp_is_full(self):
        sig = Signal(np.ones(1000), sample_rate=1000)
        faded = sig.fade_in(0.5)  # 500-sample ramp
        # Sample at index 499 (last fade sample) should be multiplied by 1.0
        assert faded.data[499] == pytest.approx(1.0)

    def test_fade_in_beyond_signal_length_no_error(self):
        sig = Signal(np.ones(10), sample_rate=10)
        out = sig.fade_in(999.0)  # duration >> signal length
        assert isinstance(out, Signal)

    def test_fade_out_last_sample_is_zero(self):
        sig = Signal(np.ones(1000), sample_rate=1000)
        faded = sig.fade_out(0.5)
        assert faded.data[-1] == pytest.approx(0.0)

    def test_fade_out_beyond_signal_length_no_error(self):
        sig = Signal(np.ones(10), sample_rate=10)
        out = sig.fade_out(999.0)
        assert isinstance(out, Signal)

    def test_fade_in_out_chain(self):
        sig = Signal.sine(440, duration=1.0, sample_rate=8000)
        out = sig.fade_in(0.05).fade_out(0.05)
        assert isinstance(out, Signal)
        assert len(out) == len(sig)


class TestBandstop:
    def test_bandstop_attenuates_stopped_frequency(self):
        t = np.arange(8000) / 8000
        mixed = np.sin(2 * np.pi * 200 * t) + np.sin(2 * np.pi * 1000 * t)
        sig = Signal(mixed, sample_rate=8000)
        filtered = sig.bandstop(900, 1100)
        orig_mag = sig.fft().in_range(950, 1050).peak_magnitude
        filt_mag = filtered.fft().in_range(950, 1050).peak_magnitude
        assert filt_mag < orig_mag * 0.1  # at least 10x attenuation

    def test_bandstop_passes_outside_frequencies(self):
        t = np.arange(8000) / 8000
        mixed = np.sin(2 * np.pi * 200 * t) + np.sin(2 * np.pi * 1000 * t)
        sig = Signal(mixed, sample_rate=8000)
        filtered = sig.bandstop(900, 1100)
        orig_low = sig.fft().in_range(150, 250).peak_magnitude
        filt_low = filtered.fft().in_range(150, 250).peak_magnitude
        assert filt_low > orig_low * 0.9  # 200 Hz component largely preserved

    def test_bandstop_cutoff_out_of_range_raises(self):
        sig = Signal.sine(440, duration=0.1, sample_rate=8000)
        with pytest.raises(ValueError, match="out of range"):
            sig.bandstop(900, 5000)  # 5000 Hz > Nyquist (4000 Hz)

    def test_bandstop_returns_signal(self):
        sig = Signal.sine(440, duration=0.1, sample_rate=8000)
        assert isinstance(sig.bandstop(300, 600), Signal)


class TestWindow:
    """Tests for Signal.window() (applying a taper)."""

    def test_window_tapers_ends_to_zero(self):
        sig = Signal(np.ones(1000), sample_rate=1000)
        out = sig.window("hann")
        assert out.data[0] == pytest.approx(0.0, abs=1e-9)
        assert out.data[-1] == pytest.approx(0.0, abs=1e-9)

    def test_window_preserves_length_and_rate(self):
        sig = Signal.sine(440, duration=0.1, sample_rate=8000)
        out = sig.window("hamming")
        assert len(out) == len(sig)
        assert out.sample_rate == 8000

    def test_window_returns_new_signal(self):
        sig = Signal(np.ones(10), sample_rate=10)
        assert sig.window() is not sig

    def test_window_default_is_hann(self):
        sig = Signal.sine(440, duration=0.1, sample_rate=8000)
        np.testing.assert_array_equal(sig.window().data, sig.window("hann").data)

    @pytest.mark.parametrize("win", ["hann", "hamming", "blackman", "bartlett"])
    def test_all_windows_run(self, win):
        sig = Signal.sine(440, duration=0.1, sample_rate=8000)
        assert isinstance(sig.window(win), Signal)

    def test_window_case_insensitive(self):
        sig = Signal.sine(440, duration=0.1, sample_rate=8000)
        assert isinstance(sig.window("HANN"), Signal)

    def test_unknown_window_raises(self):
        sig = Signal.sine(440, duration=0.1, sample_rate=8000)
        with pytest.raises(ValueError, match="Unknown window"):
            sig.window("kaiser")

    def test_window_then_fft_preserves_peak_frequency(self):
        """sig.window(w).fft() peaks at the same frequency as fft(window=w)."""
        sig = Signal.sine(440, duration=1.0, sample_rate=8000)
        assert sig.window("hann").fft().peak_frequency == pytest.approx(440, abs=2)

    def test_window_then_fft_is_not_amplitude_equivalent(self):
        """window().fft() lacks the coherent-gain correction that fft(window=)
        applies, so its magnitude is scaled down by the window's mean (~0.5 for
        Hann). This guards against re-introducing a false 'equivalence' claim."""
        sig = Signal.sine(440, duration=1.0, sample_rate=8000, amplitude=1.0)
        corrected = sig.fft(window="hann").peak_magnitude
        plain = sig.window("hann").fft().peak_magnitude
        w = np.hanning(len(sig))
        assert plain == pytest.approx(corrected * w.mean(), rel=1e-3)
        assert corrected == pytest.approx(1.0, abs=0.05)


class TestConvolve:
    """Tests for Signal.convolve()."""

    def test_matches_numpy_full(self):
        sig = Signal(np.array([1.0, 2.0, 3.0]), sample_rate=3)
        kernel = np.array([0.0, 1.0, 0.5])
        out = sig.convolve(kernel, mode="full")
        np.testing.assert_array_almost_equal(
            out.data, np.convolve([1.0, 2.0, 3.0], [0.0, 1.0, 0.5], mode="full")
        )

    def test_same_mode_preserves_length(self):
        sig = Signal(np.arange(10, dtype=float), sample_rate=10)
        out = sig.convolve(np.ones(3) / 3, mode="same")
        assert len(out) == len(sig)

    def test_full_mode_length(self):
        sig = Signal(np.ones(5), sample_rate=5)
        out = sig.convolve(np.ones(3), mode="full")
        assert len(out) == 5 + 3 - 1

    def test_moving_average_smooths(self):
        sig = Signal(np.array([0.0, 10.0, 0.0, 10.0, 0.0]), sample_rate=5)
        out = sig.convolve(np.ones(3) / 3, mode="same")
        # The smoothed signal should have lower peak-to-peak spread
        assert np.ptp(out.data) < np.ptp(sig.data)

    def test_accepts_signal_argument(self):
        a = Signal(np.array([1.0, 2.0, 3.0]), sample_rate=4)
        b = Signal(np.array([1.0, 0.0]), sample_rate=4)
        out = a.convolve(b)
        assert isinstance(out, Signal)
        assert out.sample_rate == 4

    def test_preserves_sample_rate(self):
        sig = Signal(np.ones(5), sample_rate=8000)
        assert sig.convolve(np.ones(3)).sample_rate == 8000

    def test_mismatched_sample_rate_raises(self):
        a = Signal(np.ones(5), sample_rate=8000)
        b = Signal(np.ones(3), sample_rate=44100)
        with pytest.raises(ValueError, match="sample rate"):
            a.convolve(b)

    def test_non_1d_kernel_raises(self):
        sig = Signal(np.ones(5), sample_rate=5)
        with pytest.raises(ValueError, match="1-D"):
            sig.convolve(np.ones((3, 3)))


class TestCorrelate:
    """Tests for Signal.correlate() and Signal.time_delay()."""

    def test_correlate_returns_signal(self):
        a = Signal(np.array([1.0, 2.0, 3.0]), sample_rate=3)
        b = Signal(np.array([0.0, 1.0, 0.5]), sample_rate=3)
        assert isinstance(a.correlate(b), Signal)

    def test_correlate_full_length(self):
        a = Signal(np.ones(5), sample_rate=5)
        b = Signal(np.ones(3), sample_rate=5)
        assert len(a.correlate(b)) == 5 + 3 - 1

    def test_correlate_matches_scipy(self):
        from scipy.signal import correlate as sp_correlate

        a = Signal(np.array([1.0, 2.0, 3.0, 4.0]), sample_rate=4)
        b = Signal(np.array([0.0, 1.0, 0.5]), sample_rate=4)
        np.testing.assert_array_almost_equal(
            a.correlate(b).data, sp_correlate(a.data, b.data, mode="full")
        )

    def test_time_delay_positive_when_self_lags(self):
        sr = 1000
        rng = np.random.default_rng(0)
        base = rng.standard_normal(500)
        delayed = np.concatenate([np.zeros(100), base[:-100]])  # 0.1 s later
        a = Signal(delayed, sample_rate=sr)
        b = Signal(base, sample_rate=sr)
        assert a.time_delay(b) == pytest.approx(0.1, abs=1e-6)

    def test_time_delay_sign_is_antisymmetric(self):
        sr = 1000
        rng = np.random.default_rng(1)
        base = rng.standard_normal(500)
        delayed = np.concatenate([np.zeros(50), base[:-50]])
        a = Signal(delayed, sample_rate=sr)
        b = Signal(base, sample_rate=sr)
        assert a.time_delay(b) == pytest.approx(-b.time_delay(a))

    def test_time_delay_identical_signals_is_zero(self):
        sig = Signal.noise(duration=0.5, sample_rate=1000, seed=7)
        assert sig.time_delay(sig) == pytest.approx(0.0)

    def test_time_delay_accepts_array(self):
        sr = 1000
        rng = np.random.default_rng(2)
        base = rng.standard_normal(300)
        a = Signal(np.concatenate([np.zeros(30), base[:-30]]), sample_rate=sr)
        assert a.time_delay(base) == pytest.approx(0.03, abs=1e-6)

    def test_mismatched_sample_rate_raises(self):
        a = Signal(np.ones(5), sample_rate=8000)
        b = Signal(np.ones(3), sample_rate=44100)
        with pytest.raises(ValueError, match="sample rate"):
            a.time_delay(b)


class TestFindPeaks:
    def test_finds_peak_times(self):
        # Three pulses at 0.1 s, 0.3 s and 0.6 s
        sr = 1000
        data = np.zeros(1000)
        data[[100, 300, 600]] = 1.0
        sig = Signal(data, sample_rate=sr)
        peaks = sig.find_peaks()
        np.testing.assert_allclose(peaks.times, [0.1, 0.3, 0.6])

    def test_reports_peak_heights(self):
        sr = 1000
        data = np.zeros(500)
        data[100] = 0.8
        data[300] = 0.4
        sig = Signal(data, sample_rate=sr)
        peaks = sig.find_peaks()
        np.testing.assert_allclose(peaks.times, [0.1, 0.3])
        np.testing.assert_allclose(peaks.heights, [0.8, 0.4])

    def test_result_unpacks_and_has_length(self):
        sr = 1000
        data = np.zeros(500)
        data[[100, 300]] = 1.0
        sig = Signal(data, sample_rate=sr)
        result = sig.find_peaks()
        assert len(result) == 2
        times, heights = result
        np.testing.assert_allclose(times, [0.1, 0.3])
        np.testing.assert_allclose(heights, [1.0, 1.0])

    def test_min_height_filters_small_peaks(self):
        sr = 1000
        data = np.zeros(500)
        data[100] = 1.0
        data[300] = 0.2  # below threshold
        sig = Signal(data, sample_rate=sr)
        peaks = sig.find_peaks(min_height=0.5)
        np.testing.assert_allclose(peaks.times, [0.1])

    def test_min_distance_merges_close_peaks(self):
        sr = 1000
        data = np.zeros(500)
        data[100] = 1.0
        data[110] = 0.9  # 0.01 s after the first, lower
        sig = Signal(data, sample_rate=sr)
        # Without a distance constraint, both are returned.
        assert len(sig.find_peaks()) == 2
        # With a 0.05 s minimum spacing, only the taller one survives.
        peaks = sig.find_peaks(min_distance=0.05)
        np.testing.assert_allclose(peaks.times, [0.1])

    def test_returns_empty_for_flat_signal(self):
        sig = Signal(np.zeros(100), sample_rate=100)
        peaks = sig.find_peaks()
        assert len(peaks) == 0
        assert len(peaks.times) == 0
        assert len(peaks.heights) == 0

    def test_negative_min_distance_raises(self):
        sig = Signal(np.zeros(100), sample_rate=100)
        with pytest.raises(ValueError, match="non-negative"):
            sig.find_peaks(min_distance=-0.1)
