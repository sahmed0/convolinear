"""Tests for Signal.spectrogram() and the Spectrogram class."""

import numpy as np
import pytest

from convolinear import Signal, Spectrogram


class TestSpectrogram:
    """Tests for Signal.spectrogram() and the Spectrogram class."""

    def test_returns_spectrogram(self):
        sig = Signal.sine(1000, duration=1.0, sample_rate=8000)
        assert isinstance(sig.spectrogram(), Spectrogram)

    def test_shape_consistency(self):
        sig = Signal.sine(1000, duration=1.0, sample_rate=8000)
        spec = sig.spectrogram(segment_length=256)
        assert spec.magnitudes.shape == (len(spec.frequencies), len(spec.times))
        assert spec.shape == spec.magnitudes.shape
        assert len(spec) == len(spec.times)

    def test_frequency_axis_spans_to_nyquist(self):
        sig = Signal.sine(1000, duration=1.0, sample_rate=8000)
        spec = sig.spectrogram()
        assert spec.frequencies[0] == pytest.approx(0.0)
        assert spec.frequencies[-1] == pytest.approx(4000.0)

    def test_detects_tone_frequency(self):
        sig = Signal.sine(1000, duration=1.0, sample_rate=8000)
        spec = sig.spectrogram(segment_length=512)
        dominant = np.median(spec.peak_frequency_over_time())
        assert dominant == pytest.approx(1000, abs=20)

    def test_magnitude_scaling_matches_fft_convention(self):
        """A unit-amplitude sine should read magnitude ~1 (like Signal.fft)."""
        sig = Signal.sine(1000, duration=1.0, sample_rate=8000, amplitude=1.0)
        spec = sig.spectrogram(segment_length=512)
        assert float(spec.magnitudes.max()) == pytest.approx(1.0, abs=0.1)

    def test_tracks_chirp_upward(self):
        """A rising chirp's dominant frequency should increase over time."""
        sig = Signal.from_function(
            lambda t: np.sin(2 * np.pi * (200 + 800 * t) * t),
            duration=2.0,
            sample_rate=8000,
        )
        pf = sig.spectrogram(segment_length=256).peak_frequency_over_time()
        # Compare the average of the first quarter to the last quarter
        q = len(pf) // 4
        assert pf[:q].mean() < pf[-q:].mean()

    def test_segment_longer_than_signal_is_clamped(self):
        sig = Signal.sine(1000, duration=0.01, sample_rate=8000)  # 80 samples
        spec = sig.spectrogram(segment_length=1024)
        assert isinstance(spec, Spectrogram)

    def test_unknown_window_raises(self):
        sig = Signal.sine(1000, duration=0.1, sample_rate=8000)
        with pytest.raises(ValueError, match="Unknown window"):
            sig.spectrogram(window="kaiser")

    @pytest.mark.parametrize("bad_overlap", [-0.1, 1.0, 1.5])
    def test_invalid_overlap_raises(self, bad_overlap):
        sig = Signal.sine(1000, duration=0.1, sample_rate=8000)
        with pytest.raises(ValueError, match="overlap"):
            sig.spectrogram(overlap=bad_overlap)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            Spectrogram(
                frequencies=np.array([0.0, 1.0]),
                times=np.array([0.0, 1.0, 2.0]),
                magnitudes=np.zeros((2, 2)),  # wrong: should be (2, 3)
            )

    def test_repr(self):
        sig = Signal.sine(1000, duration=1.0, sample_rate=8000)
        text = repr(sig.spectrogram())
        assert "Spectrogram" in text and "Hz" in text
