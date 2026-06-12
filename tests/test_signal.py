"""Tests for convolinear."""

import numpy as np
import pytest
from convolinear import Signal, Spectrum, Spectrogram


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
        path = self._write_csv(
            tmp_path, "t,value\n0.0,1.0\n0.01,2.0\n0.02,3.0\n0.03,4.0\n"
        )
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

        df = pd.DataFrame(
            {
                "ts": [
                    "2024-01-01 00:00:00.000",
                    "2024-01-01 00:00:00.010",
                    "2024-01-01 00:00:00.020",
                    "2024-01-01 00:00:00.030",
                ],
                "v": [1.0, 2.0, 3.0, 4.0],
            }
        )
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
    def _write_audio(
        self, tmp_path, data: np.ndarray, sample_rate: int, filename: str
    ) -> str:
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

        df = pd.DataFrame(
            {
                "ts": [
                    "2024-01-01 00:00:00.000",
                    "2024-01-01 00:00:00.010",
                    "2024-01-01 00:00:00.020",
                    "2024-01-01 00:00:00.030",
                ],
                "value": [1.0, 2.0, 3.0, 4.0],
            }
        )
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


class TestToDataframe:
    def test_default_columns(self):
        sig = Signal(np.array([0.5, -0.5, 0.25]), sample_rate=10)
        df = sig.to_dataframe()
        assert list(df.columns) == ["time", "amplitude"]
        np.testing.assert_array_almost_equal(df["amplitude"].to_numpy(), sig.data)
        np.testing.assert_array_almost_equal(df["time"].to_numpy(), sig.time_axis)

    def test_custom_column_names(self):
        sig = Signal.sine(440, duration=0.1, sample_rate=8000)
        df = sig.to_dataframe(value_column="volts", time_column="t_s")
        assert list(df.columns) == ["t_s", "volts"]

    def test_omit_time_column(self):
        sig = Signal(np.array([1.0, 2.0, 3.0]), sample_rate=10)
        df = sig.to_dataframe(time_column=None)
        assert list(df.columns) == ["amplitude"]

    def test_row_count_matches_signal(self):
        sig = Signal.sine(440, duration=0.1, sample_rate=8000)
        assert len(sig.to_dataframe()) == len(sig)

    def test_roundtrip_through_from_pandas(self):
        sig = Signal.sine(440, duration=0.1, sample_rate=8000)
        df = sig.to_dataframe(time_index=True)
        recovered = Signal.from_pandas(df, column="amplitude")
        assert recovered.sample_rate == sig.sample_rate
        np.testing.assert_array_almost_equal(recovered.data, sig.data)

    def test_time_index_named(self):
        sig = Signal(np.array([1.0, 2.0, 3.0]), sample_rate=10)
        df = sig.to_dataframe(time_index=True)
        assert df.index.name == "time"
        np.testing.assert_array_almost_equal(df.index.to_numpy(), sig.time_axis)


class TestToNumpy:
    def test_default_returns_samples(self):
        sig = Signal(np.array([0.5, -0.5, 0.25]), sample_rate=10)
        arr = sig.to_numpy()
        assert arr.ndim == 1
        np.testing.assert_array_almost_equal(arr, sig.data)

    def test_default_is_a_copy(self):
        sig = Signal(np.array([1.0, 2.0, 3.0]), sample_rate=10)
        arr = sig.to_numpy()
        arr[0] = 999.0
        assert sig.data[0] == 1.0

    def test_copy_false_returns_view(self):
        sig = Signal(np.array([1.0, 2.0, 3.0]), sample_rate=10)
        arr = sig.to_numpy(copy=False)
        assert arr is sig.data

    def test_include_time_shape(self):
        sig = Signal(np.array([0.5, -0.5, 0.25]), sample_rate=10)
        arr = sig.to_numpy(include_time=True)
        assert arr.shape == (3, 2)
        np.testing.assert_array_almost_equal(arr[:, 0], sig.time_axis)
        np.testing.assert_array_almost_equal(arr[:, 1], sig.data)

    def test_roundtrip_through_from_numpy(self):
        sig = Signal.sine(440, duration=0.1, sample_rate=8000)
        arr = sig.to_numpy(include_time=True)
        recovered = Signal.from_numpy(arr, sample_rate=sig.sample_rate, column=1)
        np.testing.assert_array_almost_equal(recovered.data, sig.data)


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
        assert filt_mag < orig_mag * 0.1  # at least 10× attenuation

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
