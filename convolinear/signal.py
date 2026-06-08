"""Core Signal class for time-domain signal manipulation."""

from __future__ import annotations
from typing import Callable, Optional, Union, TYPE_CHECKING
import numpy as np
from scipy import signal as scipy_signal
from scipy.io import wavfile, loadmat

# imports Spectrum only at type-check time so fft() doesn't create a circular import
if TYPE_CHECKING:
    from .spectrum import Spectrum


class Signal:
    """A time-domain signal: an array of samples with a known sample rate.

    Signals are immutable. Each transformation returns a new Signal,
    allowing fluent method chaining:

        Signal.from_wav("audio.wav").normalize().bandpass(300, 3000).plot()
    """

    def __init__(self, data: np.ndarray, sample_rate: int):
        """NOTE: Treat Signal.data as read-only.
        May add self.data.flags.writeable = False in the future
        To enforce immutability. """
        self.data = np.asarray(data, dtype=np.float64)
        self.sample_rate = int(sample_rate)

        if self.data.ndim != 1:
            raise ValueError(
                f"Signal data must be 1D (mono), got shape {self.data.shape}. "
                "For stereo, take a single channel."
            )
        if self.sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {sample_rate}")

    # --- Constructors ---

    @classmethod
    def from_wav(cls, path: str) -> "Signal":
        """Load a signal from a WAV file. Stereo files are converted to mono."""
        sample_rate, raw = wavfile.read(path)
        original_dtype = raw.dtype
        data = raw.astype(np.float64)
        if data.ndim > 1:
            data = data.mean(axis=1)  # average channels to mono
        if np.issubdtype(original_dtype, np.integer):
            data = data / np.iinfo(original_dtype).max
        return cls(data, sample_rate)

    @classmethod
    def from_audio(cls, path: str) -> "Signal":
        """Load a signal from an audio file (FLAC, MP3, OGG, and others)."""
        try:
            import soundfile as sf
        except ImportError:
            raise ImportError(
                "from_audio() requires the 'soundfile' package. "
                "Install it with: pip install soundfile "
                "MP3 support depends on libsndfile being compiled with MP3 support. "
            )
        data, sample_rate = sf.read(path)
        data = data.astype(np.float64)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return cls(data, sample_rate)

    @classmethod
    def _rate_from_dataframe(
        cls,
        df,
        time_column: Optional[str],
        sample_rate: Optional[int],
    ) -> int:
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
                f"Time column '{time_column}' not found. "
                f"Available columns: {available}"
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

        elapsed = elapsed.to_numpy(dtype=np.float64)

        if len(elapsed) < 2:
            raise ValueError("Need at least 2 rows to infer sample rate.")

        # Infer sample rate from the median interval between samples.
        # Using the median (rather than start/end) is more robust to gaps or
        # jitter in sensor logs.
        intervals = np.diff(elapsed)
        if np.any(intervals <= 0):
            raise ValueError(
                "Timestamps must be strictly increasing. "
                "Sort the data by time before loading."
            )

        median_interval = float(np.median(intervals))
        inferred_rate = round(1.0 / median_interval)
        return inferred_rate


    @classmethod
    def from_csv(
        cls,
        path: str,
        value_column: str,
        time_column: Optional[str] = None,
        sample_rate: Optional[int] = None,
        **pandas_kwargs,
    ) -> "Signal":
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
            sample_rate:    Samples per second. Required if ``time_column`` is
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
                "Reading CSV files requires pandas. "
                "Install it with: pip install pandas "
            )

        df = pd.read_csv(path, **pandas_kwargs)

        if value_column not in df.columns:
            available = ", ".join(df.columns.tolist())
            raise ValueError(
                f"Column '{value_column}' not found. "
                f"Available columns: {available}"
            )

        data = df[value_column].to_numpy(dtype=np.float64)

        rate = cls._rate_from_dataframe(df, time_column, sample_rate)
        return cls(data, rate)

    @classmethod
    def from_parquet(
        cls,
        path: str,
        value_column: str,
        time_column: Optional[str] = None,
        sample_rate: Optional[int] = None,
        **pandas_kwargs,
    ) -> "Signal":
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
            sample_rate:    Samples per second. Required if ``time_column`` is
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
            )

        df = pd.read_parquet(path, **pandas_kwargs)

        if value_column not in df.columns:
            available = ", ".join(df.columns.tolist())
            raise ValueError(
                f"Column '{value_column}' not found. "
                f"Available columns: {available}"
            )

        data = df[value_column].to_numpy(dtype=np.float64)

        rate = cls._rate_from_dataframe(df, time_column, sample_rate)
        return cls(data, rate)

    @classmethod
    def from_numpy(
        cls,
        array: Union[np.ndarray, str],
        sample_rate: int,
        column: Optional[int] = None,
    ) -> "Signal":
        """Load a signal from a NumPy array or a .npy / .npz file.

        Args:
            array:       A 1D NumPy array of samples, or a path string to a
                         .npy or .npz file.
            sample_rate: Samples per second.
            column:      If ``array`` is 2D (multiple channels), which column
                         index to use. Defaults to None (resolve to 0 for 2D arrays).

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
                    f"Column index {col} is out of range for array with "
                    f"{array.shape[1]} columns."
                )
            array = array[:, col]
        elif array.ndim != 1:
            raise ValueError(
                f"Array must be 1D or 2D, got shape {array.shape}."
            )

        return cls(array, sample_rate)

    @classmethod
    def from_pandas(
        cls,
        series_or_df,
        sample_rate: Optional[int] = None,
        column: Optional[str] = None,
    ) -> "Signal":
        """Load a signal from a pandas Series or DataFrame.

        For a Series, values are used directly.
        For a DataFrame, ``column`` specifies which column to use.

        If the Series or DataFrame index is a DatetimeIndex or a numeric index
        representing seconds, the sample rate will be inferred automatically.
        An explicit ``sample_rate`` always takes precedence.

        Args:
            series_or_df: A ``pandas.Series`` or ``pandas.DataFrame``.
            sample_rate:  Samples per second. Inferred from the index if absent.
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
            )

        if isinstance(series_or_df, pd.DataFrame):
            if column is None:
                raise ValueError(
                    "When passing a DataFrame you must specify 'column', "
                    "e.g. Signal.from_pandas(df, column='voltage')."
                )
            if column not in series_or_df.columns:
                available = ", ".join(series_or_df.columns.tolist())
                raise ValueError(
                    f"Column '{column}' not found. Available: {available}"
                )
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
            inferred_rate = round(1.0 / float(intervals.median()))
            return cls(data, inferred_rate)

        if pd.api.types.is_numeric_dtype(index):
            if len(index) < 2:
                raise ValueError("Need at least 2 rows to infer sample rate.")
            intervals = np.diff(index.to_numpy(dtype=np.float64))
            inferred_rate = round(1.0 / float(np.median(intervals)))
            return cls(data, inferred_rate)

        raise ValueError(
            "Could not infer sample rate from index. "
            "Please provide an explicit 'sample_rate'."
        )

    @classmethod
    def from_matlab(
        cls,
        path: str,
        variable: Optional[str] = None,
        sample_rate_variable: Optional[str] = None,
        sample_rate: Optional[int] = None,
        column: Optional[int] = None,
    ) -> "Signal":
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
            sample_rate:           Explicit sample rate. Takes precedence over
                                   ``sample_rate_variable``.
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
                sample_rate = int(np.asarray(mat[sample_rate_variable]).flat[0])
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
                    f"Variable '{variable}' not found. "
                    f"Available variables: {', '.join(data_keys)}"
                )
            array = np.asarray(mat[variable], dtype=np.float64).squeeze()
        else:
            # Auto-detect: find numeric arrays (exclude scalars / sample rate)
            numeric_keys = [
                k for k in data_keys
                if isinstance(mat[k], np.ndarray) and mat[k].size > 1
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
                    f"Column index {col} is out of range for array with "
                    f"{array.shape[1]} columns."
                )
            array = array[:, col]
        elif array.ndim != 1:
            raise ValueError(
                f"Expected a 1D or 2D array from the .mat file, "
                f"got shape {array.shape}."
            )

        return cls(array, sample_rate)

    @classmethod
    def from_function(
        cls,
        func: Callable[[np.ndarray], np.ndarray],
        duration: float,
        sample_rate: int = 44100,
    ) -> "Signal":
        """Generate a signal by sampling a function of time.

        Example:
            sine = Signal.from_function(lambda t: np.sin(2*np.pi*440*t), duration=1.0)
        """
        n_samples = int(duration * sample_rate)
        t = np.arange(n_samples) / sample_rate
        data = func(t)
        return cls(data, sample_rate)

    @classmethod
    def sine(
        cls, frequency: float, duration: float, sample_rate: int = 44100,
        amplitude: float = 1.0, phase: float = 0.0,
    ) -> "Signal":
        """Generate a pure sine wave. Convenience constructor."""
        return cls.from_function(
            lambda t: amplitude * np.sin(2 * np.pi * frequency * t + phase),
            duration,
            sample_rate,
        )

    @classmethod
    def noise(
        cls, duration: float, sample_rate: int = 44100, amplitude: float = 1.0,
        seed: Optional[int] = None,
    ) -> "Signal":
        """Generate white noise. Convenience constructor."""
        rng = np.random.default_rng(seed)
        n_samples = int(duration * sample_rate)
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

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return (
            f"Signal(samples={len(self.data)}, "
            f"sample_rate={self.sample_rate} Hz, "
            f"duration={self.duration:.3f} s)"
        )

    # --- Transformations (each returns a new Signal) ---

    def normalize(self) -> "Signal":
        """Scale the signal so its peak absolute value is 1.0."""
        peak = np.max(np.abs(self.data))
        if peak == 0:
            return Signal(self.data.copy(), self.sample_rate)
        return Signal(self.data / peak, self.sample_rate)

    def trim(self, start: float = 0.0, end: Optional[float] = None) -> "Signal":
        """Cut the signal to a time range, in seconds."""
        start_idx = int(start * self.sample_rate)
        end_idx = int(end * self.sample_rate) if end is not None else len(self.data)
        return Signal(self.data[start_idx:end_idx], self.sample_rate)

    def gain(self, factor: float) -> "Signal":
        """Multiply the signal by a constant (in linear scale)."""
        return Signal(self.data * factor, self.sample_rate)

    def gain_db(self, db: float) -> "Signal":
        """Apply gain in decibels. +6dB doubles amplitude, -6dB halves it."""
        return self.gain(10 ** (db / 20))

    def lowpass(self, cutoff: float, order: int = 4) -> "Signal":
        """Apply a Butterworth low-pass filter. Cutoff in Hz."""
        return self._butter_filter(cutoff, order, btype="low")

    def highpass(self, cutoff: float, order: int = 4) -> "Signal":
        """Apply a Butterworth high-pass filter. Cutoff in Hz."""
        return self._butter_filter(cutoff, order, btype="high")

    def bandpass(self, low: float, high: float, order: int = 4) -> "Signal":
        """Apply a Butterworth band-pass filter. Frequencies in Hz."""
        return self._butter_filter([low, high], order, btype="band")

    def _butter_filter(
        self, cutoff: Union[float, list], order: int, btype: str
    ) -> "Signal":
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
    
    def __add__(self, other: "Signal") -> "Signal":
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

    def resample(self, new_sample_rate: int) -> "Signal":
        """Resample the signal to a new sample rate."""
        new_n = int(len(self.data) * new_sample_rate / self.sample_rate)
        resampled = np.asarray(scipy_signal.resample(self.data, new_n))
        return Signal(resampled, new_sample_rate)

    # NOTE: Spectrum doesn't need quotes due to if TYPE_CHECKING import
    def fft(self) -> Spectrum:
        """Convert to the frequency domain via FFT.

        Returns a Spectrum object with magnitude vs frequency.
        """
        from .spectrum import Spectrum

        n = len(self.data)
        frequencies = np.fft.rfftfreq(n, d=1 / self.sample_rate)
        # Normalise so amplitudes are physically meaningful
        magnitudes = np.abs(np.fft.rfft(self.data)) * 2 / n
        return Spectrum(magnitudes, frequencies)

    # --- Input/Output & visualisation ---

    def to_wav(self, path: str) -> "Signal":
        """Save the signal as a 16-bit WAV file. Returns self for chaining."""
        # Clip to [-1, 1] and convert to int16
        clipped = np.clip(self.data, -1.0, 1.0)
        int_data = (clipped * 32767).astype(np.int16)
        wavfile.write(path, self.sample_rate, int_data)
        return self

    def plot(self, title: Optional[str] = None, xlabel: Optional[str] = None, ylabel: Optional[str] = None, ax=None):
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
