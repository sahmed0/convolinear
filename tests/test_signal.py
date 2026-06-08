"""Tests for convolinear."""

import numpy as np
import pytest
from convolinear import Signal, Spectrum


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
            Signal.sine(1000, duration=0.5, sample_rate=8000)
            .gain(0.5)
            .normalize()
            .trim(0.1, 0.4)
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


# ─── New constructor tests ────────────────────────────────────────────────────

class TestFromCSV:
    def _write_csv(self, tmp_path, rows: str) -> str:
        p = tmp_path / "test.csv"
        p.write_text(rows)
        return str(p)

    def test_explicit_sample_rate(self, tmp_path):
        path = self._write_csv(tmp_path, "value\n1.0\n2.0\n3.0\n4.0\n")
        sig = Signal.from_csv(path, value_column="value", sample_rate=100)
        assert len(sig) == 4
        assert sig.sample_rate == 100
        np.testing.assert_array_almost_equal(sig.data, [1.0, 2.0, 3.0, 4.0])

    def test_infer_rate_from_numeric_time_column(self, tmp_path):
        path = self._write_csv(tmp_path, "t,value\n0.0,1.0\n0.01,2.0\n0.02,3.0\n0.03,4.0\n")
        sig = Signal.from_csv(path, value_column="value", time_column="t")
        assert sig.sample_rate == 100

    def test_infer_rate_from_datetime_column(self, tmp_path):
        path = self._write_csv(
            tmp_path,
            "ts,value\n"
            "2024-01-01 00:00:00.000,1.0\n"
            "2024-01-01 00:00:00.010,2.0\n"
            "2024-01-01 00:00:00.020,3.0\n"
            "2024-01-01 00:00:00.030,4.0\n",
        )
        sig = Signal.from_csv(path, value_column="value", time_column="ts")
        assert sig.sample_rate == 100

    def test_missing_value_column_raises(self, tmp_path):
        path = self._write_csv(tmp_path, "v\n1.0\n2.0\n")
        with pytest.raises(ValueError, match="not found"):
            Signal.from_csv(path, value_column="nonexistent", sample_rate=10)

    def test_no_rate_and_no_time_column_raises(self, tmp_path):
        path = self._write_csv(tmp_path, "value\n1.0\n2.0\n")
        with pytest.raises(ValueError, match="sample_rate"):
            Signal.from_csv(path, value_column="value")


class TestFromNumpy:
    def test_from_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        sig = Signal.from_numpy(arr, sample_rate=10)
        np.testing.assert_array_almost_equal(sig.data, arr)
        assert sig.sample_rate == 10

    def test_from_npy_file(self, tmp_path):
        arr = np.array([0.5, -0.5, 1.0, -1.0])
        path = str(tmp_path / "data.npy")
        np.save(path, arr)
        sig = Signal.from_numpy(path, sample_rate=50)
        np.testing.assert_array_almost_equal(sig.data, arr)

    def test_from_npz_file(self, tmp_path):
        arr = np.array([1.0, 2.0, 3.0])
        path = str(tmp_path / "data.npz")
        np.savez(path, signal=arr)
        sig = Signal.from_numpy(path, sample_rate=50)
        np.testing.assert_array_almost_equal(sig.data, arr)

    def test_2d_array_selects_column(self):
        arr = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        sig = Signal.from_numpy(arr, sample_rate=10, column=1)
        np.testing.assert_array_almost_equal(sig.data, [10.0, 20.0, 30.0])

    def test_2d_array_out_of_range_raises(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="out of range"):
            Signal.from_numpy(arr, sample_rate=10, column=5)


class TestFromPandas:
    def test_from_series_explicit_rate(self):
        import pandas as pd
        series = pd.Series([1.0, 2.0, 3.0, 4.0])
        sig = Signal.from_pandas(series, sample_rate=100)
        assert sig.sample_rate == 100
        np.testing.assert_array_almost_equal(sig.data, [1.0, 2.0, 3.0, 4.0])

    def test_from_dataframe_with_column(self):
        import pandas as pd
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        sig = Signal.from_pandas(df, column="b", sample_rate=10)
        np.testing.assert_array_almost_equal(sig.data, [4.0, 5.0, 6.0])

    def test_dataframe_without_column_raises(self):
        import pandas as pd
        df = pd.DataFrame({"a": [1.0, 2.0]})
        with pytest.raises(ValueError, match="column"):
            Signal.from_pandas(df, sample_rate=10)

    def test_infer_rate_from_datetime_index(self):
        import pandas as pd
        index = pd.date_range("2024-01-01", periods=4, freq="10ms")
        series = pd.Series([1.0, 2.0, 3.0, 4.0], index=index)
        sig = Signal.from_pandas(series)
        assert sig.sample_rate == 100

    def test_infer_rate_from_numeric_index(self):
        import pandas as pd
        series = pd.Series([1.0, 2.0, 3.0, 4.0], index=[0.0, 0.01, 0.02, 0.03])
        sig = Signal.from_pandas(series)
        assert sig.sample_rate == 100

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError, match="Series or DataFrame"):
            Signal.from_pandas([1.0, 2.0, 3.0], sample_rate=10)


class TestFromMatlab:
    def _make_mat(self, tmp_path, data: dict) -> str:
        from scipy.io import savemat
        path = str(tmp_path / "test.mat")
        savemat(path, data)
        return path

    def test_explicit_variable_and_rate(self, tmp_path):
        path = self._make_mat(tmp_path, {"sig": np.array([1.0, 2.0, 3.0])})
        s = Signal.from_matlab(path, variable="sig", sample_rate=100)
        np.testing.assert_array_almost_equal(s.data, [1.0, 2.0, 3.0])
        assert s.sample_rate == 100

    def test_sample_rate_from_variable(self, tmp_path):
        path = self._make_mat(tmp_path, {"sig": np.array([1.0, 2.0, 3.0]), "Fs": 500})
        s = Signal.from_matlab(path, variable="sig", sample_rate_variable="Fs")
        assert s.sample_rate == 500

    def test_auto_detect_single_variable(self, tmp_path):
        path = self._make_mat(tmp_path, {"my_signal": np.array([1.0, 2.0, 3.0])})
        s = Signal.from_matlab(path, sample_rate=100)
        np.testing.assert_array_almost_equal(s.data, [1.0, 2.0, 3.0])

    def test_ambiguous_variables_raises(self, tmp_path):
        path = self._make_mat(
            tmp_path, {"sig1": np.array([1.0, 2.0]), "sig2": np.array([3.0, 4.0])}
        )
        with pytest.raises(ValueError, match="Multiple"):
            Signal.from_matlab(path, sample_rate=100)

    def test_missing_variable_raises(self, tmp_path):
        path = self._make_mat(tmp_path, {"sig": np.array([1.0, 2.0])})
        with pytest.raises(ValueError, match="not found"):
            Signal.from_matlab(path, variable="nonexistent", sample_rate=100)

    def test_no_sample_rate_raises(self, tmp_path):
        path = self._make_mat(tmp_path, {"sig": np.array([1.0, 2.0])})
        with pytest.raises(ValueError, match="sample_rate"):
            Signal.from_matlab(path, variable="sig")

    def test_multichannel_selects_column(self, tmp_path):
        arr = np.column_stack([np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0])])
        path = self._make_mat(tmp_path, {"data": arr})
        s = Signal.from_matlab(path, variable="data", sample_rate=100, column=1)
        np.testing.assert_array_almost_equal(s.data, [10.0, 20.0, 30.0])


class TestRateFromDataframe:
    def test_explicit_sample_rate_takes_precedence(self):
        import pandas as pd
        df = pd.DataFrame({"t": [0.0, 0.01, 0.02, 0.03], "v": [1.0, 2.0, 3.0, 4.0]})
        rate = Signal._rate_from_dataframe(df, time_column="t", sample_rate=500)
        assert rate == 500

    def test_infer_from_numeric_time_column(self):
        import pandas as pd
        df = pd.DataFrame({"t": [0.0, 0.01, 0.02, 0.03], "v": [1.0, 2.0, 3.0, 4.0]})
        rate = Signal._rate_from_dataframe(df, time_column="t", sample_rate=None)
        assert rate == 100

    def test_infer_from_datetime_string_column(self):
        import pandas as pd
        df = pd.DataFrame({
            "ts": [
                "2024-01-01 00:00:00.000",
                "2024-01-01 00:00:00.010",
                "2024-01-01 00:00:00.020",
                "2024-01-01 00:00:00.030",
            ],
            "v": [1.0, 2.0, 3.0, 4.0],
        })
        rate = Signal._rate_from_dataframe(df, time_column="ts", sample_rate=None)
        assert rate == 100

    def test_missing_time_column_raises(self):
        import pandas as pd
        df = pd.DataFrame({"v": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="not found"):
            Signal._rate_from_dataframe(df, time_column="nonexistent", sample_rate=None)

    def test_no_rate_no_time_column_raises(self):
        import pandas as pd
        df = pd.DataFrame({"v": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="sample_rate"):
            Signal._rate_from_dataframe(df, time_column=None, sample_rate=None)

    def test_single_row_raises(self):
        import pandas as pd
        df = pd.DataFrame({"t": [0.0], "v": [1.0]})
        with pytest.raises(ValueError):
            Signal._rate_from_dataframe(df, time_column="t", sample_rate=None)

    def test_non_increasing_timestamps_raises(self):
        import pandas as pd
        df = pd.DataFrame({"t": [0.0, 0.02, 0.01, 0.03], "v": [1.0, 2.0, 3.0, 4.0]})
        with pytest.raises(ValueError, match="increasing"):
            Signal._rate_from_dataframe(df, time_column="t", sample_rate=None)


class TestFromAudio:
    def _write_audio(self, tmp_path, data: np.ndarray, sample_rate: int, filename: str) -> str:
        import soundfile as sf
        path = str(tmp_path / filename)
        # Use FLOAT subtype for WAV to avoid 16-bit PCM quantization error
        fmt = "WAV" if filename.endswith(".wav") else None
        subtype = "FLOAT" if filename.endswith(".wav") else None
        sf.write(path, data, sample_rate, format=fmt, subtype=subtype)
        return path

    def test_loads_mono_wav(self, tmp_path):
        data = np.array([0.5, -0.5, 1.0, -1.0])
        path = self._write_audio(tmp_path, data, 8000, "mono.wav")
        sig = Signal.from_audio(path)
        assert sig.sample_rate == 8000
        np.testing.assert_array_almost_equal(sig.data, data)

    def test_stereo_to_mono(self, tmp_path):
        ch1 = np.array([1.0, 0.0, -1.0, 0.0])
        ch2 = np.array([0.0, 1.0, 0.0, -1.0])
        stereo = np.column_stack([ch1, ch2])
        path = self._write_audio(tmp_path, stereo, 8000, "stereo.wav")
        sig = Signal.from_audio(path)
        expected = (ch1 + ch2) / 2
        np.testing.assert_array_almost_equal(sig.data, expected)

    def test_loads_flac_file(self, tmp_path):
        data = np.array([0.1, 0.2, 0.3, -0.1])
        path = self._write_audio(tmp_path, data, 44100, "test.flac")
        sig = Signal.from_audio(path)
        assert sig.sample_rate == 44100
        np.testing.assert_array_almost_equal(sig.data, data, decimal=5)

    def test_returns_signal_instance(self, tmp_path):
        data = np.zeros(100)
        path = self._write_audio(tmp_path, data, 1000, "zeros.wav")
        sig = Signal.from_audio(path)
        assert isinstance(sig, Signal)


class TestFromParquet:
    def _write_parquet(self, tmp_path, df) -> str:
        path = str(tmp_path / "test.parquet")
        df.to_parquet(path)
        return path

    def test_explicit_sample_rate(self, tmp_path):
        import pandas as pd
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0]})
        path = self._write_parquet(tmp_path, df)
        sig = Signal.from_parquet(path, value_column="value", sample_rate=100)
        assert sig.sample_rate == 100
        np.testing.assert_array_almost_equal(sig.data, [1.0, 2.0, 3.0, 4.0])

    def test_infer_rate_from_numeric_time_column(self, tmp_path):
        import pandas as pd
        df = pd.DataFrame({"t": [0.0, 0.01, 0.02, 0.03], "value": [1.0, 2.0, 3.0, 4.0]})
        path = self._write_parquet(tmp_path, df)
        sig = Signal.from_parquet(path, value_column="value", time_column="t")
        assert sig.sample_rate == 100

    def test_infer_rate_from_datetime_column(self, tmp_path):
        import pandas as pd
        df = pd.DataFrame({
            "ts": [
                "2024-01-01 00:00:00.000",
                "2024-01-01 00:00:00.010",
                "2024-01-01 00:00:00.020",
                "2024-01-01 00:00:00.030",
            ],
            "value": [1.0, 2.0, 3.0, 4.0],
        })
        path = self._write_parquet(tmp_path, df)
        sig = Signal.from_parquet(path, value_column="value", time_column="ts")
        assert sig.sample_rate == 100

    def test_missing_value_column_raises(self, tmp_path):
        import pandas as pd
        df = pd.DataFrame({"value": [1.0, 2.0]})
        path = self._write_parquet(tmp_path, df)
        with pytest.raises(ValueError, match="not found"):
            Signal.from_parquet(path, value_column="nonexistent", sample_rate=10)

    def test_no_rate_no_time_column_raises(self, tmp_path):
        import pandas as pd
        df = pd.DataFrame({"value": [1.0, 2.0]})
        path = self._write_parquet(tmp_path, df)
        with pytest.raises(ValueError, match="sample_rate"):
            Signal.from_parquet(path, value_column="value")


class TestFromWav:
    def test_loads_mono_wav(self, tmp_path):
        from scipy.io import wavfile
        data = np.array([0.5, -0.5, 0.25, -0.25], dtype=np.float32)
        path = str(tmp_path / "mono.wav")
        wavfile.write(path, 8000, data)
        sig = Signal.from_wav(path)
        assert sig.sample_rate == 8000
        np.testing.assert_array_almost_equal(sig.data, data, decimal=5)

    def test_stereo_to_mono(self, tmp_path):
        from scipy.io import wavfile
        ch1 = np.array([1.0, 0.0, -1.0, 0.0], dtype=np.float32)
        ch2 = np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float32)
        stereo = np.column_stack([ch1, ch2])
        path = str(tmp_path / "stereo.wav")
        wavfile.write(path, 8000, stereo)
        sig = Signal.from_wav(path)
        assert len(sig) == 4
        np.testing.assert_array_almost_equal(sig.data, (ch1 + ch2) / 2, decimal=5)

    def test_integer_normalization(self, tmp_path):
        from scipy.io import wavfile
        # Max int16 value should normalize to 1.0
        data = np.array([32767, -32768, 16384], dtype=np.int16)
        path = str(tmp_path / "int16.wav")
        wavfile.write(path, 44100, data)
        sig = Signal.from_wav(path)
        assert sig.data[0] == pytest.approx(1.0, rel=1e-4)
        assert sig.data[1] == pytest.approx(-1.0, rel=1e-4)


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


class TestToWav:
    def test_roundtrip_sample_rate(self, tmp_path):
        from scipy.io import wavfile
        sig = Signal.sine(440, duration=0.1, sample_rate=8000)
        path = str(tmp_path / "out.wav")
        sig.to_wav(path)
        sr, _ = wavfile.read(path)
        assert sr == 8000

    def test_returns_self(self, tmp_path):
        sig = Signal.sine(440, duration=0.1, sample_rate=8000)
        path = str(tmp_path / "out.wav")
        result = sig.to_wav(path)
        assert result is sig

    def test_data_approximately_preserved(self, tmp_path):
        from scipy.io import wavfile
        sig = Signal(np.array([0.5, -0.5, 0.25, -0.25]), sample_rate=8000)
        path = str(tmp_path / "out.wav")
        sig.to_wav(path)
        _, raw = wavfile.read(path)
        recovered = raw.astype(np.float64) / 32767
        np.testing.assert_array_almost_equal(recovered, sig.data, decimal=3)


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