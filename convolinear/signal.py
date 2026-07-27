"""Core Signal class for time-domain signal manipulation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import numpy.typing as npt
from scipy import signal as scipy_signal
from scipy.io import loadmat, wavfile


@dataclass(frozen=True)
class PeakResult:
    """The peaks found by :meth:`Signal.find_peaks`.

    Holds two parallel arrays - the ``times`` (in seconds) at which peaks
    occur and their ``heights`` (the signal amplitude at each peak), both in
    ascending time order.

    Unpacks as ``times, heights = sig.find_peaks(...)`` and reports the number
    of peaks via ``len(result)``.
    """

    times: np.ndarray
    heights: np.ndarray

    def __iter__(self) -> Iterator[np.ndarray]:
        yield self.times
        yield self.heights

    def __len__(self) -> int:
        return len(self.times)


# imports Spectrum/Spectrogram only at type-check time so fft()/spectrogram()
# don't create a circular import
if TYPE_CHECKING:
    from .spectrogram import Spectrogram
    from .spectrum import Spectrum


class Signal:
    """A time-domain signal: an array of samples with a known sample rate.

    Signals are immutable. Each transformation returns a new Signal,
    allowing fluent method chaining:

        Signal.from_wav("audio.wav").normalize().bandpass(300, 3000).plot()
    """

    def __init__(self, data: npt.ArrayLike, sample_rate: float):
        """Create a Signal from a 1-D array of samples and a sample rate in Hz.

        Args:
            data:        A 1-D array of samples (mono). The array is copied and
                         frozen, so the Signal is immutable and never aliases a
                         caller-owned array.
            sample_rate: Samples per second. A float that may be fractional or
                         below 1 Hz - e.g. ``1 / 86400`` for daily data.

        Raises:
            ValueError: if ``data`` is not 1-D, is empty, or ``sample_rate`` is
                not finite and positive.
        """
        arr = np.asarray(data, dtype=np.float64)
        if arr is data:
            # np.asarray returned the caller's own object (already float64) -
            # copy so freezing doesn't mutate an array the caller still holds.
            arr = arr.copy()

        if arr.ndim != 1:
            raise ValueError(
                f"Signal data must be 1-D (mono), got shape {arr.shape}. "
                "For stereo, take a single channel."
            )
        if arr.size == 0:
            raise ValueError("Signal must contain at least one sample; got an empty array.")

        self._sample_rate = float(sample_rate)
        if not math.isfinite(self._sample_rate) or self._sample_rate <= 0:
            raise ValueError(f"Sample rate must be a finite positive number, got {sample_rate}.")

        arr.flags.writeable = False
        self._data = arr

    @property
    def data(self) -> npt.NDArray[np.float64]:
        """The signal's samples as a read-only 1-D array.

        Writing to the returned array raises ``ValueError``; use
        :meth:`to_numpy` (which copies by default) for a writable array.
        """
        return self._data

    @property
    def sample_rate(self) -> float:
        """The sample rate in Hz (read-only)."""
        return self._sample_rate

    # --- Constructors ---

    @classmethod
    def from_wav(cls, path: str) -> Signal:
        """Load a signal from a WAV file. Stereo files are converted to mono."""
        sample_rate, raw = wavfile.read(path)
        original_dtype = raw.dtype
        data = raw.astype(np.float64)
        if data.ndim > 1:
            data = data.mean(axis=1)  # average channels to mono
        if np.issubdtype(original_dtype, np.unsignedinteger):
            # 8-bit WAV is unsigned offset-binary: silence = 128, full scale = [0, 255].
            info = np.iinfo(original_dtype)
            half = (info.max + 1) / 2.0
            data = (data - half) / half
        elif np.issubdtype(original_dtype, np.integer):
            # Divide by |min| (e.g. 32768), not max (32767): maps the full int range
            # into [-1, 1) exactly, instead of pushing the minimum below -1.
            data = data / -float(np.iinfo(original_dtype).min)
        return cls(data, sample_rate)

    @classmethod
    def from_audio(cls, path: str) -> Signal:
        """Load a signal from an audio file (WAV,FLAC, MP3, OGG, and others)."""
        try:
            import soundfile as sf
        except ImportError:
            raise ImportError(
                "from_audio() requires the 'soundfile' package. "
                "Install it with: pip install soundfile "
                "MP3 support depends on libsndfile being compiled with MP3 support. "
            ) from None
        data, sample_rate = sf.read(path)
        data = data.astype(np.float64)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return cls(data, sample_rate)

    @classmethod
    def _rate_from_dataframe(
        cls,
        df: pd.DataFrame,
        time_column: str | None,
        sample_rate: float | None,
    ) -> float:
        """Shared logic for inferring sample rate from Dataframe time column."""

        import pandas as pd

        if sample_rate is not None:
            return sample_rate

        if time_column is None:
            raise ValueError(
                "You must provide either a 'time_column' (so the sample rate "
                "can be inferred) or an explicit 'sample_rate'."
            )

        if time_column not in df.columns:
            available = ", ".join(df.columns.tolist())
            raise ValueError(
                f"Time column '{time_column}' not found. Available columns: {available}"
            )

        times = df[time_column]

        # Handle datetime strings by converting to seconds since the first sample.
        # NOTE: Check for object/string dtypes rather than using np.issubdtype, which
        # is incompatible with newer pandas StringDtype.
        is_numeric = pd.api.types.is_numeric_dtype(times)
        if not is_numeric:
            times = pd.to_datetime(times)
            elapsed = (times - times.iloc[0]).dt.total_seconds()
        else:
            elapsed = times - times.iloc[0]

        elapsed_arr = elapsed.to_numpy(dtype=np.float64)

        if len(elapsed_arr) < 2:
            raise ValueError("Need at least 2 rows to infer sample rate.")

        # Infer sample rate from the median interval between samples.
        # Using the median (rather than start/end) is more robust to gaps or
        # jitter in sensor logs.
        intervals = np.diff(elapsed_arr)
        if np.any(intervals <= 0):
            raise ValueError(
                "Timestamps must be strictly increasing. Sort the data by time before loading."
            )

        median_interval = float(np.median(intervals))
        # Return the raw float rate (no rounding): daily data, for example,
        # has a true rate of 1/86400 Hz that rounding would collapse to zero.
        return 1.0 / median_interval

    @classmethod
    def from_csv(
        cls,
        path: str,
        value_column: str,
        time_column: str | None = None,
        sample_rate: float | None = None,
        **pandas_kwargs: Any,
    ) -> Self:
        """Load a signal from a CSV file.

        You must provide either a ``time_column`` (and the sample rate will be
        inferred from it) or an explicit ``sample_rate``. If you provide both,
        the explicit ``sample_rate`` takes precedence.

        Args:
            path:           Path to the CSV file.
            value_column:   Name of the column containing the signal values.
            time_column:    Name of the column containing timestamps. Can be
                            numeric seconds or a datetime string - both are
                            handled automatically.
            sample_rate:    Samples per second (a float; fractional and sub-1 Hz
                            rates are supported). Required if ``time_column`` is
                            not provided.
            **pandas_kwargs: Extra keyword arguments forwarded to
                            ``pandas.read_csv`` (e.g. ``sep``, ``skiprows``).

        Example::

            # CSV with a numeric time column - sample rate inferred
            Signal.from_csv("sensor.csv", value_column="voltage", time_column="t_s")

            # CSV with no time column - sample rate provided explicitly
            Signal.from_csv("samples.csv", value_column="pressure", sample_rate=500)

            # CSV with ISO datetime timestamps
            Signal.from_csv("log.csv", value_column="temp", time_column="timestamp")
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "Reading CSV files requires pandas. Install it with: pip install pandas "
            ) from None

        df = pd.read_csv(path, **pandas_kwargs)

        if value_column not in df.columns:
            available = ", ".join(df.columns.tolist())
            raise ValueError(f"Column '{value_column}' not found. Available columns: {available}")

        data = df[value_column].to_numpy(dtype=np.float64)

        rate = cls._rate_from_dataframe(df, time_column, sample_rate)
        return cls(data, rate)

    @classmethod
    def from_parquet(
        cls,
        path: str,
        value_column: str,
        time_column: str | None = None,
        sample_rate: float | None = None,
        **pandas_kwargs: Any,
    ) -> Self:
        """Load a signal from a Parquet file.

        You must provide either a ``time_column`` (and the sample rate will be
        inferred from it) or an explicit ``sample_rate``. If you provide both,
        the explicit ``sample_rate`` takes precedence.

        Args:
            path:           Path to the parquet file.
            value_column:   Name of the column containing the signal values.
            time_column:    Name of the column containing timestamps. Can be
                            numeric seconds or a datetime string - both are
                            handled automatically.
            sample_rate:    Samples per second (a float; fractional and sub-1 Hz
                            rates are supported). Required if ``time_column`` is
                            not provided.
            **pandas_kwargs: Extra keyword arguments forwarded to
                            ``pandas.read_parquet`` (e.g. ``columns``, ``filters``).

        Example::

            # Parquet with a numeric time column - sample rate inferred
            Signal.from_parquet("sensor.parquet", value_column="voltage", time_column="t_s")

            # Parquet with no time column - sample rate provided explicitly
            Signal.from_parquet("samples.parquet", value_column="pressure", sample_rate=500)

            # Parquet with ISO datetime timestamps
            Signal.from_parquet("log.parquet", value_column="temp", time_column="timestamp")
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "Reading parquet files requires pandas and pyarrow. "
                "Install them with: pip install pandas pyarrow "
            ) from None

        df = pd.read_parquet(path, **pandas_kwargs)

        if value_column not in df.columns:
            available = ", ".join(df.columns.tolist())
            raise ValueError(f"Column '{value_column}' not found. Available columns: {available}")

        data = df[value_column].to_numpy(dtype=np.float64)

        rate = cls._rate_from_dataframe(df, time_column, sample_rate)
        return cls(data, rate)

    @classmethod
    def from_numpy(
        cls,
        array: np.ndarray | str,
        sample_rate: float,
        column: int | None = None,
    ) -> Self:
        """Load a signal from a NumPy array or a .npy / .npz file.

        Args:
            array:       A 1D NumPy array of samples, or a path string to a
                         .npy or .npz file.
            sample_rate: Samples per second (a float; fractional and sub-1 Hz
                         rates are supported).
            column:      If ``array`` is 2D (multiple channels), which column
                         index to use. Defaults to None (resolved to 0 for 2D arrays).

        Example::

            # From an array already in memory
            Signal.from_numpy(my_array, sample_rate=1000)

            # From a saved .npy file
            Signal.from_numpy("data.npy", sample_rate=500)

            # From a .npz archive - first array is used by default
            Signal.from_numpy("data.npz", sample_rate=500)

            # Multi-channel array - pick channel 1
            Signal.from_numpy(multichannel_array, sample_rate=1000, column=1)
        """
        if isinstance(array, str):
            path = array
            if path.endswith(".npz"):
                archive = np.load(path)
                keys = list(archive.keys())
                if not keys:
                    raise ValueError(f".npz archive '{path}' contains no arrays.")
                array = archive[keys[0]]
            else:
                array = np.load(path)

        array = np.asarray(array, dtype=np.float64)

        if array.ndim == 2:
            col = column if column is not None else 0
            if col >= array.shape[1]:
                raise ValueError(
                    f"Column index {col} is out of range for array with {array.shape[1]} columns."
                )
            array = array[:, col]
        elif array.ndim != 1:
            raise ValueError(f"Array must be 1D or 2D, got shape {array.shape}.")

        return cls(array, sample_rate)

    @classmethod
    def from_pandas(
        cls,
        series_or_df: pd.Series[float] | pd.DataFrame,
        sample_rate: float | None = None,
        column: str | None = None,
    ) -> Self:
        """Load a signal from a pandas Series or DataFrame.

        For a Series, values are used directly.
        For a DataFrame, ``column`` specifies which column to use.

        If the Series or DataFrame index is a DatetimeIndex or a numeric index
        representing seconds, the sample rate will be inferred automatically.
        An explicit ``sample_rate`` always takes precedence.

        Args:
            series_or_df: A ``pandas.Series`` or ``pandas.DataFrame``.
            sample_rate:  Samples per second (a float; fractional and sub-1 Hz
                          rates are supported). Inferred from the index if absent.
            column:       Column name to use when passing a DataFrame.

        Example::

            import pandas as pd

            # From a Series
            Signal.from_pandas(df["voltage"], sample_rate=1000)

            # From a DataFrame - column name required
            Signal.from_pandas(df, column="voltage", sample_rate=1000)

            # With a DatetimeIndex - sample rate inferred automatically
            Signal.from_pandas(df["voltage"])
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "from_pandas requires pandas. Install it with: pip install pandas"
            ) from None

        if isinstance(series_or_df, pd.DataFrame):
            if column is None:
                raise ValueError(
                    "When passing a DataFrame you must specify 'column', "
                    "e.g. Signal.from_pandas(df, column='voltage')."
                )
            if column not in series_or_df.columns:
                available = ", ".join(series_or_df.columns.tolist())
                raise ValueError(f"Column '{column}' not found. Available: {available}")
            series = series_or_df[column]
        elif isinstance(series_or_df, pd.Series):
            series = series_or_df
        else:
            raise TypeError(
                f"Expected a pandas Series or DataFrame, got {type(series_or_df).__name__}."
            )

        data = series.to_numpy(dtype=np.float64)

        if sample_rate is not None:
            return cls(data, sample_rate)

        # Try to infer sample rate from the index
        index = series.index
        if isinstance(index, pd.DatetimeIndex):
            if len(index) < 2:
                raise ValueError("Need at least 2 rows to infer sample rate.")
            intervals = pd.Series(index).diff().dropna().dt.total_seconds()
            return cls(data, 1.0 / float(intervals.median()))

        if pd.api.types.is_numeric_dtype(index):
            if len(index) < 2:
                raise ValueError("Need at least 2 rows to infer sample rate.")
            interval_arr = np.diff(index.to_numpy(dtype=np.float64))
            return cls(data, 1.0 / float(np.median(interval_arr)))

        raise ValueError(
            "Could not infer sample rate from index. Please provide an explicit 'sample_rate'."
        )

    @classmethod
    def from_matlab(
        cls,
        path: str,
        variable: str | None = None,
        sample_rate_variable: str | None = None,
        sample_rate: float | None = None,
        column: int | None = None,
    ) -> Self:
        """Load a signal from a MATLAB .mat file.

        Supports MATLAB files up to v7.2 (the large majority of .mat files).
        For v7.3 files (saved with ``-v7.3`` in MATLAB, actually HDF5 format),
        A ``from_hdf5`` constructor may be implemented in the future.

        Args:
            path:                  Path to the .mat file.
            variable:              Name of the variable to load. If the file
                                   contains only one numeric variable, it is
                                   selected automatically.
            sample_rate_variable:  Name of a scalar variable in the file that
                                   holds the sample rate (e.g. ``"fs"`` or
                                   ``"Fs"``). Common in MATLAB workspaces.
            sample_rate:           Explicit sample rate (a float; fractional and
                                   sub-1 Hz rates are supported). Takes precedence
                                   over ``sample_rate_variable``.
            column:                For multi-column arrays, which column index
                                   to use. Defaults to 0.

        Example::

            # MATLAB file with one signal array and a sample rate variable
            Signal.from_matlab("eeg.mat", variable="eeg", sample_rate_variable="Fs")

            # Explicit sample rate
            Signal.from_matlab("data.mat", variable="accel_x", sample_rate=2048)

            # Auto-detect the only numeric variable in the file
            Signal.from_matlab("simple.mat", sample_rate=100)
        """
        mat = loadmat(path)

        # Filter out metadata keys that loadmat injects starting with '__'
        data_keys = [k for k in mat.keys() if not k.startswith("__")]

        # Resolve sample rate: explicit arg > variable in file > error
        if sample_rate is None:
            if sample_rate_variable is not None:
                if sample_rate_variable not in mat:
                    raise ValueError(
                        f"Sample rate variable '{sample_rate_variable}' not found. "
                        f"Available variables: {', '.join(data_keys)}"
                    )
                sample_rate = float(np.asarray(mat[sample_rate_variable]).flat[0])
            else:
                raise ValueError(
                    "Provide either 'sample_rate' or 'sample_rate_variable' "
                    "(the name of the variable in the .mat file that holds the "
                    "sample rate, e.g. sample_rate_variable='Fs')."
                )

        # Resolve which variable holds the signal
        if variable is not None:
            if variable not in mat:
                raise ValueError(
                    f"Variable '{variable}' not found. Available variables: {', '.join(data_keys)}"
                )
            array = np.asarray(mat[variable], dtype=np.float64).squeeze()
        else:
            # Auto-detect: find numeric arrays (exclude scalars / sample rate)
            numeric_keys = [
                k for k in data_keys if isinstance(mat[k], np.ndarray) and mat[k].size > 1
            ]
            if len(numeric_keys) == 0:
                raise ValueError(
                    "No numeric array variables found in the .mat file. "
                    f"Available variables: {', '.join(data_keys)}"
                )
            if len(numeric_keys) > 1:
                raise ValueError(
                    f"Multiple numeric variables found: {', '.join(numeric_keys)}. "
                    "Specify which one to use with the 'variable' argument."
                )
            array = np.asarray(mat[numeric_keys[0]], dtype=np.float64).squeeze()

        # Handle multi-column arrays (e.g. multi-channel (stereo) recordings)
        if array.ndim == 2:
            col = column if column is not None else 0
            if col >= array.shape[1]:
                raise ValueError(
                    f"Column index {col} is out of range for array with {array.shape[1]} columns."
                )
            array = array[:, col]
        elif array.ndim != 1:
            raise ValueError(
                f"Expected a 1D or 2D array from the .mat file, got shape {array.shape}."
            )

        return cls(array, sample_rate)

    @classmethod
    def from_function(
        cls,
        func: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        duration: float,
        sample_rate: float = 44100.0,
    ) -> Self:
        """Generate a signal by sampling a function of time.

        ``sample_rate`` is a float (fractional and sub-1 Hz rates are supported).
        ``duration * sample_rate`` must round to at least one sample, otherwise
        the resulting empty signal is rejected at construction.

        Example::
            sine = Signal.from_function(lambda t: np.sin(2*np.pi*440*t), duration=1.0)
        """
        n_samples = round(duration * sample_rate)
        t = np.arange(n_samples) / sample_rate
        data = func(t)
        return cls(data, sample_rate)

    @classmethod
    def sine(
        cls,
        frequency: float,
        duration: float,
        sample_rate: float = 44100.0,
        amplitude: float = 1.0,
        phase: float = 0.0,
    ) -> Self:
        """Generate a pure sine wave. Convenience constructor."""
        return cls.from_function(
            lambda t: amplitude * np.sin(2 * np.pi * frequency * t + phase),
            duration,
            sample_rate,
        )

    @classmethod
    def noise(
        cls,
        duration: float,
        sample_rate: float = 44100.0,
        amplitude: float = 1.0,
        seed: int | None = None,
    ) -> Self:
        """Generate white noise. Convenience constructor."""
        rng = np.random.default_rng(seed)
        n_samples = round(duration * sample_rate)
        return cls(amplitude * rng.standard_normal(n_samples), sample_rate)

    # --- Properties ---

    @property
    def duration(self) -> float:
        """Length of the signal in seconds."""
        return len(self.data) / self.sample_rate

    @property
    def time_axis(self) -> np.ndarray:
        """Array of time values for each sample, in seconds."""
        return np.arange(len(self.data)) / self.sample_rate

    @property
    def rms(self) -> float:
        """Root mean square amplitude - a measure of average signal energy.

        A full-scale sine wave (amplitude 1.0) has an RMS of 1/√2 ≈ 0.707.
        """
        return float(np.sqrt(np.mean(self.data**2)))

    @property
    def rms_db(self) -> float:
        """RMS amplitude in dBFS (decibels relative to full scale).

        Returns ``-inf`` for a silent signal.
        """
        r = self.rms
        return float("-inf") if r == 0.0 else float(20 * np.log10(r))

    @property
    def power(self) -> float:
        """Mean square power - the average of the squared samples.

        This is the square of :attr:`rms`. A full-scale sine wave
        (amplitude 1.0) has a power of 0.5.
        """
        return float(np.mean(self.data**2))

    @property
    def power_db(self) -> float:
        """Mean square power in decibels (relative to full scale).

        Equal to :attr:`rms_db`, since power is the square of RMS.
        Returns ``-inf`` for a silent signal.
        """
        p = self.power
        return float("-inf") if p == 0.0 else float(10 * np.log10(p))

    @property
    def peak_db(self) -> float:
        """Peak absolute amplitude in dBFS.

        Returns ``-inf`` for a silent signal.
        """
        p = float(np.max(np.abs(self.data)))
        return float("-inf") if p == 0.0 else float(20 * np.log10(p))

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return (
            f"Signal(samples={len(self.data)}, "
            f"sample_rate={self.sample_rate:g} Hz, "
            f"duration={self.duration:.3f} s)"
        )

    # --- Transformations (each returns a new Signal) ---

    def normalize(self) -> Signal:
        """Scale the signal so its peak absolute value is 1.0."""
        peak = np.max(np.abs(self.data))
        if peak == 0:
            return Signal(self.data.copy(), self.sample_rate)
        return Signal(self.data / peak, self.sample_rate)

    def trim(self, start: float = 0.0, end: float | None = None) -> Signal:
        """Cut the signal to a time range, in seconds.

        ``end`` beyond the signal's duration is capped at the end of the signal.

        Raises:
            ValueError: if ``start`` is negative, ``end`` is before ``start``,
                or the selected range contains no samples.
        """
        if start < 0:
            raise ValueError(f"trim start must be non-negative, got {start}.")
        if end is not None and end < start:
            raise ValueError(f"trim end ({end}) must not be before start ({start}).")
        start_idx = int(start * self.sample_rate)
        end_idx = int(end * self.sample_rate) if end is not None else len(self.data)
        end_idx = min(end_idx, len(self.data))
        if start_idx >= end_idx:
            raise ValueError(
                f"trim range [{start}, {end}) selects no samples from a signal of "
                f"duration {self.duration:.6g} s."
            )
        return Signal(self.data[start_idx:end_idx], self.sample_rate)

    def gain(self, factor: float) -> Signal:
        """Multiply the signal by a constant (in linear scale)."""
        return Signal(self.data * factor, self.sample_rate)

    def gain_db(self, db: float) -> Signal:
        """Apply gain in decibels. +6dB doubles amplitude, -6dB halves it."""
        return self.gain(10 ** (db / 20))

    def __add__(self, other: Signal) -> Signal:
        """Mix two signals by adding their samples element-wise.

        If the signals differ in length, the shorter one is zero-padded.
        Both signals must have the same sample rate.

        Raises:
            ValueError: if sample rates differ.
        """
        if not isinstance(other, Signal):
            return NotImplemented
        if self.sample_rate != other.sample_rate:
            raise ValueError(
                f"Cannot mix signals with different sample rates: "
                f"{self.sample_rate} Hz vs {other.sample_rate} Hz."
            )
        n = max(len(self.data), len(other.data))
        a = np.pad(self.data, (0, n - len(self.data)))
        b = np.pad(other.data, (0, n - len(other.data)))
        return Signal(a + b, self.sample_rate)

    def concat(self, other: Signal) -> Signal:
        """Append another signal onto the end of this one.

        Use this to join signals end-to-end. To overlay (mix) two signals
        at the same time, use the ``+`` operator instead.

        Both signals must have the same sample rate.

        Raises:
            TypeError: if ``other`` is not a Signal.
            ValueError: if sample rates differ.
        """
        if not isinstance(other, Signal):
            raise TypeError(f"Expected a Signal, got {type(other).__name__}.")
        if self.sample_rate != other.sample_rate:
            raise ValueError(
                f"Cannot concatenate signals with different sample rates: "
                f"{self.sample_rate} Hz vs {other.sample_rate} Hz."
            )
        return Signal(np.concatenate([self.data, other.data]), self.sample_rate)

    def window(self, window: str = "hann") -> Signal:
        """Apply a window function (taper) to the signal.

        Multiplies the signal by a window that smoothly falls to zero at both
        ends (a plain ``data * window``), reducing spectral leakage.

        For spectral analysis prefer :meth:`fft` with its ``window`` argument:
        it both windows the data and corrects the magnitude for the window's
        coherent gain, so a unit-amplitude sine still reads a magnitude of 1.
        This method applies no such correction, so ``sig.window(w).fft()`` is
        **not** amplitude-equivalent to ``sig.fft(window=w)`` (the peak is
        scaled down by the window's mean). Reach for ``window()`` when you want
        the tapered *waveform* itself - removing edge clicks before
        :meth:`to_wav`, windowed-sinc FIR design, or visualising the taper.

        Args:
            window: Window name. Accepted values are ``"hann"`` (recommended
                    for general use), ``"hamming"``, ``"blackman"``, and
                    ``"bartlett"``.

        Example::

            # Taper the edges before saving, to avoid clicks
            sig.window("hann").to_wav("tapered.wav")

        Raises:
            ValueError: if ``window`` is not a recognised name.
        """
        key = Signal._validate_window(window)
        w = Signal._FFT_WINDOWS[key](len(self.data))
        return Signal(self.data * w, self.sample_rate)

    def remove_dc(self) -> Signal:
        """Remove the DC offset by subtracting the mean from every sample."""
        return Signal(self.data - np.mean(self.data), self.sample_rate)

    def reverse(self) -> Signal:
        """Flip the signal in time."""
        return Signal(self.data[::-1].copy(), self.sample_rate)

    def clip(self, min_val: float = -1.0, max_val: float = 1.0) -> Signal:
        """Clamp all samples to the range [min_val, max_val].

        Raises:
            ValueError: if min_val >= max_val.
        """
        if min_val >= max_val:
            raise ValueError(f"min_val ({min_val}) must be less than max_val ({max_val}).")
        return Signal(np.clip(self.data, min_val, max_val), self.sample_rate)

    def fade_in(self, duration: float) -> Signal:
        """Apply a linear fade-in from 0 to 1 over ``duration`` seconds.

        The ramp is capped at the full signal length so oversized durations
        do not raise an error.
        """
        n_fade = min(int(duration * self.sample_rate), len(self.data))
        envelope = self.data.copy()
        if n_fade < 1:  # ramp shorter than a sample - nothing to fade
            return Signal(envelope, self.sample_rate)
        envelope[:n_fade] *= np.linspace(0.0, 1.0, n_fade)
        return Signal(envelope, self.sample_rate)

    def fade_out(self, duration: float) -> Signal:
        """Apply a linear fade-out from 1 to 0 over ``duration`` seconds.

        The ramp is capped at the full signal length so oversized durations
        do not raise an error.
        """
        n_fade = min(int(duration * self.sample_rate), len(self.data))
        envelope = self.data.copy()
        if n_fade < 1:  # ramp shorter than a sample; envelope[-0:] would be the
            return Signal(envelope, self.sample_rate)  # whole array - guard it
        envelope[-n_fade:] *= np.linspace(1.0, 0.0, n_fade)
        return Signal(envelope, self.sample_rate)

    def lowpass(self, cutoff: float, order: int = 4) -> Signal:
        """Apply a Butterworth low-pass filter. Cutoff in Hz."""
        return self._butter_filter(cutoff, order, btype="low")

    def highpass(self, cutoff: float, order: int = 4) -> Signal:
        """Apply a Butterworth high-pass filter. Cutoff in Hz."""
        return self._butter_filter(cutoff, order, btype="high")

    def bandpass(self, low: float, high: float, order: int = 4) -> Signal:
        """Apply a Butterworth band-pass filter. Frequencies in Hz."""
        return self._butter_filter([low, high], order, btype="band")

    def bandstop(self, low: float, high: float, order: int = 4) -> Signal:
        """Apply a Butterworth band-stop (notch) filter. Frequencies in Hz.

        Attenuates the band between ``low`` and ``high`` Hz and passes
        everything outside it. The inverse of ``bandpass``.

        Raises:
            ValueError: if either cutoff is outside (0, Nyquist).
        """
        return self._butter_filter([low, high], order, btype="bandstop")

    def _butter_filter(self, cutoff: float | list, order: int, btype: str) -> Signal:
        nyquist = self.sample_rate / 2
        normalized = np.asarray(cutoff) / nyquist
        if np.any(normalized >= 1) or np.any(normalized <= 0):
            raise ValueError(
                f"Cutoff frequency {cutoff} Hz is out of range. "
                f"Must be between 0 and {nyquist} Hz (Nyquist limit)."
            )
        sos = scipy_signal.butter(order, normalized, btype=btype, output="sos")
        filtered = scipy_signal.sosfiltfilt(sos, self.data)
        return Signal(filtered, self.sample_rate)

    def resample(self, new_sample_rate: float) -> Signal:
        """Resample the signal to a new sample rate.

        Raises:
            ValueError: if the new rate is too low to keep at least one sample.
        """
        new_n = round(len(self.data) * new_sample_rate / self.sample_rate)
        if new_n < 1:
            raise ValueError(
                f"Resampling to {new_sample_rate:g} Hz leaves no samples for a signal of "
                f"{len(self.data)} samples at {self.sample_rate:g} Hz. Choose a higher rate."
            )
        resampled = np.asarray(scipy_signal.resample(self.data, new_n))
        return Signal(resampled, new_sample_rate)

    def _coerce_other(self, other: Signal | np.ndarray, op: str) -> np.ndarray:
        """Return the sample array of ``other`` for a two-signal operation.

        Accepts either another Signal (whose sample rate must match) or a raw
        1-D array (e.g. an FIR kernel), and returns it as a 1-D float64 array.
        """
        if isinstance(other, Signal):
            if self.sample_rate != other.sample_rate:
                raise ValueError(
                    f"Cannot {op} signals with different sample rates: "
                    f"{self.sample_rate} Hz vs {other.sample_rate} Hz."
                )
            return other.data
        arr = np.asarray(other, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"Expected a Signal or 1-D array for {op}, got shape {arr.shape}.")
        return arr

    def convolve(self, other: Signal | np.ndarray, mode: str = "full") -> Signal:
        """Convolve this signal with another signal or a kernel.

        Convolution is the operation behind FIR filtering: passing a kernel
        (impulse response) here applies that filter to the signal. It is
        computed efficiently in the frequency domain via
        ``scipy.signal.fftconvolve``.

        Args:
            other: Another Signal (same sample rate) or a 1-D array to use as
                   the convolution kernel.
            mode:  Output length convention, forwarded to scipy:

                   - ``"full"``  - full discrete convolution (default).
                   - ``"same"``  - same length as this signal, centred.
                   - ``"valid"`` - only fully overlapping points.

        Example::

            # Smooth a signal with a 5-tap moving-average FIR kernel
            kernel = np.ones(5) / 5
            smoothed = sig.convolve(kernel, mode="same")

        Raises:
            ValueError: if ``other`` is a Signal with a different sample rate,
                or is not 1-D.
        """
        kernel = self._coerce_other(other, "convolve")
        if mode == "valid" and len(kernel) > len(self.data):
            raise ValueError(
                f"mode='valid' requires the kernel to be no longer than the signal, "
                f"but the kernel has {len(kernel)} samples and the signal has "
                f"{len(self.data)}. The roles of signal and kernel would silently invert."
            )
        result = scipy_signal.fftconvolve(self.data, kernel, mode=mode)
        return Signal(np.asarray(result), self.sample_rate)

    def correlate(self, other: Signal | np.ndarray, mode: str = "full") -> Signal:
        """Cross-correlate this signal with another signal or array.

        Cross-correlation measures how similar two signals are as one is slid
        past the other. The result is indexed by lag; its peak indicates the
        shift at which the two align best. For the lag itself in seconds, use
        :meth:`time_delay`.

        Args:
            other: Another Signal (same sample rate) or a 1-D array.
            mode:  Output length convention forwarded to
                   ``scipy.signal.correlate`` (``"full"``, ``"same"`` or
                   ``"valid"``). Defaults to ``"full"``.

        Returns:
            A Signal holding the cross-correlation values, sharing this
            signal's sample rate.
        """
        other_data = self._coerce_other(other, "correlate")
        result = scipy_signal.correlate(self.data, other_data, mode=mode)
        return Signal(np.asarray(result), self.sample_rate)

    def time_delay(self, other: Signal | np.ndarray) -> float:
        """Estimate the time delay (in seconds) between this signal and ``other``.

        Finds the lag of the cross-correlation peak and converts it to seconds.
        A **positive** result means this signal is delayed relative to
        ``other`` (its features arrive later); a negative result means it
        arrives earlier.

        Example::

            # Recover a known 0.1 s delay between two microphones
            delay = mic_a.time_delay(mic_b)

        Raises:
            ValueError: if ``other`` is a Signal with a different sample rate,
                or is not 1-D.
        """
        other_data = self._coerce_other(other, "time_delay")
        corr = scipy_signal.correlate(self.data, other_data, mode="full")
        lags = scipy_signal.correlation_lags(len(self.data), len(other_data), mode="full")
        lag = int(lags[np.argmax(corr)])
        return lag / self.sample_rate

    def find_peaks(
        self,
        min_height: float | None = None,
        min_distance: float | None = None,
    ) -> PeakResult:
        """Find local maxima (peaks) in the signal.

        Useful for pulse detection, heartbeat analysis, vibration analysis and
        any task where you need to locate discrete events in a waveform.

        Args:
            min_height:   Minimum amplitude a sample must reach to count as a
                          peak. Peaks below this are ignored. Defaults to
                          ``None`` (no height threshold).
            min_distance: Minimum spacing between peaks, in seconds. When two
                          peaks fall closer than this, the lower one is
                          discarded. Useful to avoid double-counting a single
                          event. Defaults to ``None`` (no spacing constraint).

        Returns:
            A :class:`PeakResult` with the peak ``times`` (in seconds) and their
            ``heights`` (signal amplitude at each peak), in ascending time
            order. Both arrays are empty if no peaks are found.

        Example::

            # Heartbeats in an ECG, ignoring noise and ringing
            beats = ecg.find_peaks(min_height=0.5, min_distance=0.3)
            bpm = 60.0 / np.diff(beats.times).mean()

        Raises:
            ValueError: if ``min_distance`` is negative.
        """
        distance = None
        if min_distance is not None:
            if min_distance < 0:
                raise ValueError(f"min_distance must be non-negative, got {min_distance}.")
            # scipy expects the distance in samples and requires it to be >= 1.
            distance = int(max(1, round(min_distance * self.sample_rate)))

        indices, _ = scipy_signal.find_peaks(self.data, height=min_height, distance=distance)
        return PeakResult(indices / self.sample_rate, self.data[indices])

    _FFT_WINDOWS: ClassVar[dict[str, Callable[[int], np.ndarray]]] = {
        "hann": np.hanning,
        "hamming": np.hamming,
        "blackman": np.blackman,
        "bartlett": np.bartlett,
    }

    @classmethod
    def _validate_window(cls, window: str) -> str:
        """Normalise and validate a window name against the supported set.

        Returns the lower-cased key. Shared by :meth:`window`, :meth:`fft` and
        :meth:`spectrogram` so the accepted names and error message stay in
        sync across all three.
        """
        key = window.lower()
        if key not in cls._FFT_WINDOWS:
            raise ValueError(
                f"Unknown window '{window}'. Choose from: {', '.join(cls._FFT_WINDOWS)}."
            )
        return key

    # NOTE: Spectrum doesn't need quotes due to if TYPE_CHECKING import
    def fft(self, window: str | None = None) -> Spectrum:
        """Convert to the frequency domain via FFT.

        Args:
            window: Window function to reduce spectral leakage. Accepted values
                    are ``"hann"`` (recommended for general use), ``"hamming"``,
                    ``"blackman"``, and ``"bartlett"``. Defaults to ``None``
                    (rectangular window - no windowing).

        Returns a complex-valued Spectrum: it stores the raw rfft coefficients,
        so the transform is invertible via :meth:`Spectrum.to_signal`. The
        ``magnitudes`` it exposes follow the same convention as before (a
        unit-amplitude sine reads ~1).

        Raises:
            ValueError: if ``window`` is not a recognised name.
        """
        from .spectrum import Spectrum

        n = len(self.data)
        data = self.data

        if window is not None:
            key = Signal._validate_window(window)
            w = Signal._FFT_WINDOWS[key](n)
            data = data * w
            scale = 2.0 / w.sum()
        else:
            scale = 2.0 / n

        frequencies = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)
        coefficients = np.fft.rfft(data)
        # The Spectrum stores raw coefficients; the DC/Nyquist halving that keeps
        # the amplitude convention lives in Spectrum.magnitudes now.
        return Spectrum(
            coefficients,
            frequencies,
            n_samples=n,
            sample_rate=self.sample_rate,
            scale=scale,
        )

    def spectrogram(
        self,
        segment_length: int = 256,
        overlap: float = 0.5,
        window: str = "hann",
    ) -> Spectrogram:
        """Compute a spectrogram via the short-time Fourier transform (STFT).

        The signal is split into overlapping segments; each is windowed and
        Fourier-transformed, producing a picture of how the frequency content
        evolves over time. This is the standard tool for non-stationary
        signals - speech, music, chirps - whose spectrum changes as they play.

        Args:
            segment_length: Number of samples per STFT segment. Larger values
                            give finer frequency resolution but coarser time
                            resolution. Defaults to 256.
            overlap:        Fraction of overlap between consecutive segments,
                            in ``[0, 1)``. Defaults to 0.5 (50%).
            window:         Window applied to each segment. Accepted values are
                            ``"hann"`` (default), ``"hamming"``, ``"blackman"``
                            and ``"bartlett"``.

        Returns:
            A Spectrogram object with magnitude indexed by frequency and time.

        Example::

            # Visualise a frequency sweep
            chirp = Signal.from_function(
                lambda t: np.sin(2 * np.pi * (200 + 400 * t) * t), duration=2.0
            )
            chirp.spectrogram().plot(max_freq=2000)

        Raises:
            ValueError: if ``window`` is unrecognised or ``overlap`` is not in
                ``[0, 1)``.
        """
        from .spectrogram import Spectrogram

        key = Signal._validate_window(window)
        if not 0.0 <= overlap < 1.0:
            raise ValueError(f"overlap must be in [0, 1), got {overlap}.")
        if int(segment_length) < 1:
            raise ValueError(f"segment_length must be a positive integer, got {segment_length}.")

        # Never request a segment longer than the signal itself.
        nperseg = min(int(segment_length), len(self.data))
        noverlap = int(nperseg * overlap)

        frequencies, times, mags = scipy_signal.spectrogram(
            self.data,
            fs=self.sample_rate,
            window=key,
            nperseg=nperseg,
            noverlap=noverlap,
            mode="magnitude",
            scaling="spectrum",
        )
        # scipy's one-sided magnitude splits a real tone across the implied
        # +/- frequency pair; double the non-DC/Nyquist bins so a sinusoid of
        # amplitude 1 reads ~1, matching the Signal.fft() convention.
        mags = mags.copy()
        if mags.shape[0] > 1:
            last = -1 if nperseg % 2 == 0 else None
            mags[1:last] *= 2.0
        return Spectrogram(frequencies, times, mags)

    # --- Input/Output & visualisation ---

        """Save the signal as a 16-bit WAV file. Returns self for chaining.

        Raises:
            ValueError: if the sample rate is not integer-valued. WAV headers
                store an integer rate, so resample first (integer-valued floats
                such as ``48000.0`` are accepted).
        """
        if self.sample_rate != int(self.sample_rate):
            raise ValueError(
                f"WAV files require an integer sample rate; this signal's rate is "
                f"{self.sample_rate:g} Hz. Resample first, e.g. "
                f".resample({max(1, round(self.sample_rate))})."
            )
        # Clip to [-1, 1] and convert to int16
        clipped = np.clip(self.data, -1.0, 1.0)
        int_data = (clipped * 32767).astype(np.int16)
        wavfile.write(path, int(self.sample_rate), int_data)
        return self

    def to_numpy(self, include_time: bool = False, copy: bool = True) -> np.ndarray:
        """Export the signal's samples as a NumPy array.

        This is the way out of the Signal pipeline back into plain NumPy: hand
        the result to scikit-learn, PyTorch, SciPy, or any code that just wants
        an array of samples.

        Args:
            include_time: If True, return a 2D array of shape ``(n_samples, 2)``
                          whose first column is the sample times (seconds) and
                          second column is the amplitudes. Defaults to False,
                          which returns the 1D amplitude array on its own. The
                          2D form round-trips back through :meth:`from_numpy`
                          with ``column=1``.
            copy:         If True (the default), return a fresh array that is
                          safe to mutate. Pass False to get a view of the
                          underlying samples and avoid the copy, but treat the
                          result as read-only - mutating it corrupts the Signal.

        Example::

            # Drop a filtered signal into a NumPy / scikit-learn pipeline
            features = sig.bandpass(300, 3000).to_numpy()

            # Keep the time axis alongside the samples
            arr = sig.to_numpy(include_time=True)  # shape (n, 2)
            times, amplitudes = arr[:, 0], arr[:, 1]
        """
        if include_time:
            # column_stack always allocates, so the copy flag is moot here.
            return np.column_stack((self.time_axis, self.data))
        return self.data.copy() if copy else self.data

    def to_dataframe(
        self,
        value_column: str = "amplitude",
        time_column: str | None = "time",
        time_index: bool = False,
    ):
        """Export the signal to a pandas DataFrame for further analysis.

        The returned DataFrame has one row per sample, with the amplitude in
        ``value_column``. By default the sample times (in seconds) are included
        as well, so the result is self-describing and can be handed straight to
        a pandas pipeline (resampling, plotting, joining with other data, ...).

        The output round-trips back through :meth:`from_pandas`: pass
        ``time_index=True`` and the time axis becomes the DataFrame index, from
        which ``from_pandas`` can re-infer the sample rate automatically.

        Args:
            value_column: Name of the column holding the signal amplitudes.
                          Defaults to ``"amplitude"``.
            time_column:  Name of the column holding the sample times in
                          seconds. Pass ``None`` to omit the time column.
                          Ignored when ``time_index`` is True.
            time_index:   If True, place the sample times (seconds) in the
                          DataFrame index instead of a column. Defaults to
                          False (a default RangeIndex is used).

        Example::

            # Hand a filtered signal back to pandas for resampling
            df = sig.bandpass(300, 3000).to_dataframe()
            df.rolling(window=10).mean()

            # Round-trip through pandas with the sample rate preserved
            df = sig.to_dataframe(time_index=True)
            Signal.from_pandas(df, column="amplitude")

        Raises:
            ImportError: if pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "to_dataframe requires pandas. Install it with: pip install pandas"
            ) from None

        if time_index:
            return pd.DataFrame(
                {value_column: self.data},
                index=pd.Index(self.time_axis, name=time_column),
            )

        columns = {}
        if time_column is not None:
            columns[time_column] = self.time_axis
        columns[value_column] = self.data
        return pd.DataFrame(columns)

    def plot(
        self,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        ax=None,
    ):
        """Plot the signal in the time domain. Returns the matplotlib axis."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 3))
        ax.plot(self.time_axis, self.data, linewidth=0.8)
        ax.set_xlabel(xlabel or "Time (s)")
        ax.set_ylabel(ylabel or "Amplitude")
        ax.set_title(title or "Signal")
        ax.grid(True, alpha=0.3)
        return ax
