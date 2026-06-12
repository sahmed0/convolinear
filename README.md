# convolinear

A clean, chainable Python library for digital signal processing.

`convolinear` wraps the power of NumPy and SciPy in a fluent, readable API. Common DSP tasks -
loading audio, filtering, mixing, and spectral analysis - become short, expressive one-liners
instead of 10+ lines of boilerplate.

[![MIT License](https://img.shields.io/badge/MIT-2026_Sajid_Ahmed-limegreen.svg)](https://opensource.org/license/mit)
[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://www.python.org/)


### Example
You can
1. Extract data from a .WAV file,
2. Apply a bandpass filter on the data,
3. Normalise it,
4. Take the Fourier transform,
5. And plot it.

**All in a single line of code!**

```python
Signal.from_wav("audio.wav").bandpass(300, 3000).normalize().fft().plot()
```
<p align="center">
  <img src="https://github.com/sahmed0/convolinear/blob/main/demo_output.png?raw=true" alt="demo.py output graphs" width="700">
</p>

---
## Table of Contents

- [Installation](#installation)
- [Core Concepts](#core-concepts)
- [Signal - Constructor Reference](#signal--constructor-reference)
  - [`Signal(data, sample_rate)`](#signaldata-sample_rate)
  - [`Signal.from_wav(path)`](#signalfrom_wavpath)
  - [`Signal.from_audio(path)`](#signalfrom_audiopath)
  - [`Signal.from_csv(...)`](#signalfrom_csv)
  - [`Signal.from_parquet(...)`](#signalfrom_parquet)
  - [`Signal.from_numpy(array, sample_rate, column)`](#signalfrom_numpyarray-sample_rate-column)
  - [`Signal.from_pandas(series_or_df, sample_rate, column)`](#signalfrom_pandasseries_or_df-sample_rate-column)
  - [`Signal.from_matlab(...)`](#signalfrom_matlab)
  - [`Signal.sine(...)`](#signalsine)
  - [`Signal.noise(...)`](#signalnoise)
  - [`Signal.from_function(...)`](#signalfrom_function)
- [Signal - Properties](#signal--properties)
- [Signal - Transformations](#signal--transformations)
- [Signal - Filters](#signal--filters)
- [Signal - Convolution and Correlation](#signal--convolution-and-correlation)
- [Signal - I/O and Visualization](#signal--io-and-visualization)
- [Spectrum - Reference](#spectrum--reference)
- [Spectrogram - Reference](#spectrogram--reference)
- [Worked Examples](#worked-examples)
- [Development](#development)

---

## Installation

```bash
pip install convolinear
```

Install optional extras for the loaders and features you need:

```bash
pip install "convolinear[plot]"    # matplotlib - required for .plot()
pip install "convolinear[audio]"   # soundfile  - required for Signal.from_audio()
pip install "convolinear[pandas]"  # pandas + pyarrow - required for from_csv / from_parquet / from_pandas / to_dataframe
```

Combine extras in one command:

```bash
pip install "convolinear[plot,audio,pandas]"
```

**Core requirements:** Python 3.11+, NumPy ≥ 1.23.2, SciPy ≥ 1.8

| Extra | Packages installed | Unlocks |
|-------|--------------------|---------|
| `plot` | matplotlib | `Signal.plot()`, `Spectrum.plot()` |
| `audio` | soundfile | `Signal.from_audio()` |
| `pandas` | pandas, pyarrow | `Signal.from_csv()`, `Signal.from_parquet()`, `Signal.from_pandas()`, `Signal.to_dataframe()` |

> **MATLAB files** (`.mat`) are loaded via SciPy, which is already a core dependency - no extra needed.

---

## Core Concepts

### Immutability and chaining

Every transformation returns a **new** `Signal`. The original is never modified. This makes it
safe to branch from the same signal and chain operations without side effects:

```python
raw = Signal.from_wav("recording.wav")

# Two independent processing paths from the same source
voice = raw.highpass(80).bandpass(300, 3400).normalize()
full  = raw.lowpass(8000).normalize()
```

### The Signal and Spectrum types

| Type | Domain | Produced by |
|------|--------|-------------|
| `Signal` | Time (samples × amplitude) | Constructors, transformations |
| `Spectrum` | Frequency (Hz × magnitude) | `Signal.fft()` |
| `Spectrogram` | Time-frequency (Hz × time × magnitude) | `Signal.spectrogram()` |

---

## Signal - Constructor Reference

### `Signal(data, sample_rate)`

Construct a signal directly from a NumPy array.

```python
import numpy as np
from convolinear import Signal

data = np.array([0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5])
sig = Signal(data, sample_rate=8)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `np.ndarray` | 1-D array of samples (any numeric dtype, converted to float64 internally) |
| `sample_rate` | `int` | Samples per second (Hz). Must be positive. |

Raises `ValueError` if `data` is not 1-D or `sample_rate` is not positive.

---

### `Signal.from_wav(path)`

Load a signal from a WAV file.

```python
sig = Signal.from_wav("recording.wav")
```

- Stereo (or multi-channel) files are averaged to mono.
- Integer PCM samples (e.g. 16-bit) are normalised to the `[-1, 1]` range.
- Float WAV files are loaded as-is.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` | Path to the WAV file |

---

### `Signal.from_audio(path)`

Load a signal from audio files such as WAV, FLAC, MP3, OGG, and any other format supported by
the `soundfile` library.
NOTE: If you are only working with WAV files, I recommend using from_wav() as it does not need the `soundfile` package.

```python
sig = Signal.from_audio("recording.flac")
sig = Signal.from_audio("podcast.mp3")
```

- Stereo files are averaged to mono.
- Requires the `soundfile` package (`pip install "convolinear[audio]"`).

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` | Path to the audio file |

Raises `ImportError` if `soundfile` is not installed.

---

### `Signal.from_csv(...)`

Load a signal from a CSV file.

```python
# Sample rate inferred from a numeric time column
sig = Signal.from_csv("sensor.csv", value_column="voltage", time_column="time_s")

# Sample rate inferred from a datetime column
sig = Signal.from_csv("log.csv", value_column="pressure", time_column="timestamp")

# Explicit sample rate (no time column needed)
sig = Signal.from_csv("raw.csv", value_column="ch0", sample_rate=1000)

# Pass extra pandas kwargs (e.g. delimiter, skip rows)
sig = Signal.from_csv("data.tsv", value_column="amp", time_column="t",
                      sep="\t", skiprows=2)
```

You must supply either `time_column` (sample rate is inferred as the median interval) or
an explicit `sample_rate`. If you supply both, `sample_rate` wins.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | - | Path to the CSV file |
| `value_column` | `str` | - | Column containing the signal samples |
| `time_column` | `str \| None` | `None` | Column with timestamps (numeric seconds or datetime strings) |
| `sample_rate` | `int \| None` | `None` | Explicit sample rate in Hz |
| `**pandas_kwargs` | | | Extra keyword arguments forwarded to `pandas.read_csv` |

Raises:
- `ImportError` - pandas not installed
- `ValueError` - `value_column` or `time_column` not found in file
- `ValueError` - neither `time_column` nor `sample_rate` provided
- `ValueError` - timestamps are not strictly increasing
- `ValueError` - fewer than 2 rows (sample rate cannot be inferred)

---

### `Signal.from_parquet(...)`

Load a signal from a Parquet file. Identical semantics to `from_csv`.

```python
sig = Signal.from_parquet("sensor.parquet", value_column="voltage", time_column="time_s")

# Explicit sample rate
sig = Signal.from_parquet("data.parquet", value_column="ch0", sample_rate=44100)

# pandas kwargs (e.g. select only needed columns)
sig = Signal.from_parquet("big.parquet", value_column="amp", time_column="t",
                          columns=["t", "amp"])
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | - | Path to the Parquet file |
| `value_column` | `str` | - | Column containing the signal samples |
| `time_column` | `str \| None` | `None` | Column with timestamps (numeric seconds or datetime strings) |
| `sample_rate` | `int \| None` | `None` | Explicit sample rate in Hz |
| `**pandas_kwargs` | | | Extra keyword arguments forwarded to `pandas.read_parquet` |

Requires `pandas` and `pyarrow` (`pip install "convolinear[pandas]"`).

Raises the same errors as `from_csv` plus `ImportError` if `pyarrow` is missing.

---

### `Signal.from_numpy(array, sample_rate, column)`

Load a signal from a NumPy array in memory, or from a `.npy` / `.npz` file on disk.

```python
import numpy as np

# From an in-memory array
data = np.random.randn(44100)
sig = Signal.from_numpy(data, sample_rate=44100)

# From a .npy file
sig = Signal.from_numpy("recording.npy", sample_rate=8000)

# From a .npz archive - first array is used automatically
sig = Signal.from_numpy("multi.npz", sample_rate=44100)

# 2-D array (multiple channels) - pick column 1
multichannel = np.random.randn(44100, 4)
sig = Signal.from_numpy(multichannel, sample_rate=44100, column=1)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `array` | `np.ndarray \| str` | - | A 1-D or 2-D NumPy array, or a path to a `.npy` / `.npz` file |
| `sample_rate` | `int` | - | Samples per second (always required) |
| `column` | `int \| None` | `None` | Column index to use when `array` is 2-D |

Raises `ValueError` if the column index is out of range, the `.npz` archive is empty, or the
array is not 1-D or 2-D.

---

### `Signal.from_pandas(series_or_df, sample_rate, column)`

Load a signal from a pandas `Series` or `DataFrame`.

```python
import pandas as pd

# From a Series - sample rate inferred from DatetimeIndex
s = pd.Series([0.1, 0.3, -0.2], index=pd.date_range("2024-01-01", periods=3, freq="1ms"))
sig = Signal.from_pandas(s)

# From a Series with a numeric (seconds) index
s = pd.Series([0.1, 0.3, -0.2], index=[0.0, 0.001, 0.002])
sig = Signal.from_pandas(s)

# Explicit sample rate overrides inference
sig = Signal.from_pandas(s, sample_rate=1000)

# From a DataFrame - column name required
df = pd.DataFrame({"ch0": [0.1, 0.3], "ch1": [-0.2, 0.4]})
sig = Signal.from_pandas(df, column="ch0", sample_rate=1000)
```

Sample rate is inferred automatically when the index is a `DatetimeIndex` or a numeric index
representing seconds. An explicit `sample_rate` always takes precedence.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `series_or_df` | `pd.Series \| pd.DataFrame` | - | Input data |
| `sample_rate` | `int \| None` | `None` | Explicit sample rate; inferred from index if omitted |
| `column` | `str \| None` | `None` | Column name to use when passing a DataFrame (required for DataFrames) |

Raises:
- `ImportError` - pandas not installed
- `TypeError` - input is not a Series or DataFrame
- `ValueError` - DataFrame passed without `column`
- `ValueError` - `column` not found in DataFrame
- `ValueError` - fewer than 2 rows (sample rate cannot be inferred)
- `ValueError` - `sample_rate` not provided and cannot be inferred from the index

---

### `Signal.from_matlab(...)`

Load a signal from a MATLAB `.mat` file (supports files up to MATLAB format v7.2).

```python
# Auto-detect the only numeric variable in the file
sig = Signal.from_matlab("recording.mat", sample_rate=44100)

# Name the variable explicitly
sig = Signal.from_matlab("data.mat", variable="ecg", sample_rate=500)

# Read the sample rate from a scalar inside the .mat file
sig = Signal.from_matlab("data.mat", variable="signal", sample_rate_variable="fs")

# Multi-channel array - pick column 2
sig = Signal.from_matlab("eeg.mat", variable="data", sample_rate=256, column=2)
```

When the file contains exactly one numeric array, `variable` can be omitted and it is selected
automatically. If `sample_rate_variable` names a scalar variable in the file (e.g. `"fs"` or
`"Fs"`), that value is used; an explicit `sample_rate` always overrides it.

> **Note:** MATLAB v7.3 files (saved with `-v7.3`, which are actually HDF5) are not yet
> supported. Use an earlier save format from MATLAB if you encounter load errors.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | - | Path to the `.mat` file |
| `variable` | `str \| None` | `None` | Name of the variable to load; auto-selected when there is exactly one numeric array |
| `sample_rate_variable` | `str \| None` | `None` | Name of a scalar variable in the file that holds the sample rate (e.g. `"fs"`) |
| `sample_rate` | `int \| None` | `None` | Explicit sample rate; overrides `sample_rate_variable` |
| `column` | `int \| None` | `0` | Column index for multi-channel (2-D) arrays |

Raises `ValueError` for: missing/ambiguous variable, missing sample rate, column out of range,
or unsupported array shape.

---

### `Signal.sine(frequency, duration, sample_rate, amplitude, phase)`

Generate a pure sine wave.

```python
# A concert A (440 Hz) for 2 seconds
tone = Signal.sine(frequency=440, duration=2.0)

# Quieter, phase-shifted
tone2 = Signal.sine(frequency=440, duration=2.0, amplitude=0.5, phase=1.57)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `frequency` | `float` | - | Frequency in Hz |
| `duration` | `float` | - | Length in seconds |
| `sample_rate` | `int` | `44100` | Samples per second |
| `amplitude` | `float` | `1.0` | Peak amplitude |
| `phase` | `float` | `0.0` | Phase offset in radians |

---

### `Signal.noise(duration, sample_rate, amplitude, seed)`

Generate white (Gaussian) noise.

```python
noise = Signal.noise(duration=1.0, amplitude=0.1, seed=42)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `duration` | `float` | - | Length in seconds |
| `sample_rate` | `int` | `44100` | Samples per second |
| `amplitude` | `float` | `1.0` | Standard deviation of the noise |
| `seed` | `int \| None` | `None` | Random seed for reproducibility |

---

### `Signal.from_function(func, duration, sample_rate)`

Generate a signal by sampling an arbitrary function of time.

```python
import numpy as np

# Linear chirp sweeping 200 → 1000 Hz over 2 seconds
chirp = Signal.from_function(
    lambda t: np.sin(2 * np.pi * (200 + 400 * t) * t),
    duration=2.0,
    sample_rate=44100,
)

# AM-modulated tone
am = Signal.from_function(
    lambda t: (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)) * np.sin(2 * np.pi * 440 * t),
    duration=1.0,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `func` | `Callable[[np.ndarray], np.ndarray]` | - | Function that accepts a time array `t` (in seconds) and returns an array of samples |
| `duration` | `float` | - | Length in seconds |
| `sample_rate` | `int` | `44100` | Samples per second |

---

## Signal - Properties

### `sig.duration`

Length of the signal in seconds.

```python
sig = Signal.sine(440, duration=2.5)
print(sig.duration)  # 2.5
```

### `sig.sample_rate`

Samples per second.

```python
print(sig.sample_rate)  # 44100
```

### `sig.data`

The underlying data samples in the form of a `np.ndarray` of dtype `float64`.
NOTE: I recommend treating this array as read-only.

```python
print(sig.data[:10])
```

### `sig.time_axis`

NumPy array of the time value (in seconds) for each sample. Useful for plotting.

```python
import matplotlib.pyplot as plt
plt.plot(sig.time_axis, sig.data)
```

### `sig.rms`

Root mean square amplitude - a measure of the signal's average energy.

```python
tone = Signal.sine(440, duration=1.0, amplitude=1.0)
print(tone.rms)  # ≈ 0.707  (1/√2 for a full sine wave)

noise = Signal.noise(duration=1.0, amplitude=0.1)
print(noise.rms)  # ≈ 0.1
```

### `sig.rms_db`

RMS amplitude expressed in dBFS (decibels relative to full scale). Returns `-inf` for a silent signal.

```python
tone = Signal.sine(440, duration=1.0, amplitude=1.0)
print(tone.rms_db)   # ≈ -3.01  (1/√2 in dB)

silence = Signal(np.zeros(100), sample_rate=100)
print(silence.rms_db)  # -inf
```

### `sig.power`

Mean square power - the average of the squared samples, equal to the square of `sig.rms`.

```python
tone = Signal.sine(440, duration=1.0, amplitude=1.0)
print(tone.power)  # ≈ 0.5  (rms² for a full sine wave)
```

### `sig.power_db`

Mean square power expressed in decibels (relative to full scale). Equal to `sig.rms_db`,
since power is the square of RMS. Returns `-inf` for a silent signal.

```python
tone = Signal.sine(440, duration=1.0, amplitude=1.0)
print(tone.power_db)  # ≈ -3.01

silence = Signal(np.zeros(100), sample_rate=100)
print(silence.power_db)  # -inf
```

### `sig.peak_db`

Peak absolute amplitude expressed in dBFS. Returns `-inf` for a silent signal.

```python
tone = Signal.sine(440, duration=1.0, amplitude=1.0)
print(tone.peak_db)   # ≈ 0.0  (peak of 1.0 = 0 dBFS)

quiet = Signal.sine(440, duration=1.0, amplitude=0.5)
print(quiet.peak_db)  # ≈ -6.02
```

### `len(sig)`

Number of samples.

```python
sig = Signal.sine(440, duration=1.0, sample_rate=8000)
print(len(sig))  # 8000
```

### `repr(sig)`

Human-readable summary.

```python
print(Signal.sine(440, duration=1.0, sample_rate=8000))
# Signal(samples=8000, sample_rate=8000 Hz, duration=1.000 s)
```

---

## Signal - Transformations

All transformations return a new `Signal`. Chains can be as long as needed.

### `.normalize()`

Scale the signal so its peak absolute value is exactly 1.0. Silent signals (all zeros)
are returned unchanged.

```python
sig = Signal(np.array([0.0, 0.25, -0.5]), sample_rate=3)
n = sig.normalize()
# n.data → [0.0, 0.5, -1.0]
```

---

### `.trim(start, end)`

Extract a time slice. Both arguments are in seconds.

```python
sig = Signal.sine(440, duration=5.0)

# Keep only seconds 1.0 to 3.5
excerpt = sig.trim(start=1.0, end=3.5)
print(excerpt.duration)  # 2.5

# Trim just the start (keep from 0.5 s to end)
trimmed = sig.trim(start=0.5)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start` | `float` | `0.0` | Start time in seconds |
| `end` | `float \| None` | `None` | End time in seconds. `None` means the end of the signal. |

---

### `.gain(factor)`

Multiply every sample by `factor`.

```python
sig.gain(2.0)   # double the amplitude
sig.gain(0.5)   # halve the amplitude
```

---

### `.gain_db(db)`

Apply gain expressed in decibels. +6 dB ≈ ×2 amplitude; −6 dB ≈ ×0.5 amplitude.

```python
sig.gain_db(6)    # roughly double
sig.gain_db(-20)  # reduce to 10% amplitude
```

---


### `.concat(other)`

Append another signal onto the end of this one, joining them sequentially. To overlay (mix)
two signals at the same point in time, use the `+` operator instead.

Both signals must have the same sample rate.

```python
intro  = Signal.sine(440, duration=0.5)
outro  = Signal.sine(880, duration=0.5)
joined = intro.concat(outro)
print(joined.duration)  # 1.0

# Build a sequence from a list
parts = [Signal.sine(f, duration=0.25) for f in [261, 293, 329, 349]]
melody = parts[0].concat(parts[1]).concat(parts[2]).concat(parts[3])
```

Raises `ValueError` if the sample rates differ.

---

### `.resample(new_sample_rate)`

Change the sample rate. The duration stays the same; the number of samples changes.

```python
# Downsample from 44100 Hz to 8000 Hz (phone quality)
sig_44k = Signal.from_wav("audio.wav")
sig_8k = sig_44k.resample(8000)
```

---

### `sig_a + sig_b` - Mixing two signals

Add two signals together (mix them). Both must have the same sample rate.
If they have different lengths, the shorter one is zero-padded to match the longer.

```python
tone  = Signal.sine(440, duration=2.0, sample_rate=44100)
noise = Signal.noise(duration=2.0,     sample_rate=44100, amplitude=0.05)
mix   = tone + noise

# Different lengths - result is as long as the longer signal
long_tone   = Signal.sine(440, duration=2.0, sample_rate=44100)
short_noise = Signal.noise(duration=0.5, sample_rate=44100, amplitude=0.1)
mix = long_tone + short_noise  # 2.0 s result
```

Raises `ValueError` if the sample rates differ.

---

### `.window(window="hann")`

Apply a window function (taper) to the signal. A window smoothly falls to zero at both
ends, which is standard practice before an FFT to reduce **spectral leakage** - the spurious
sidelobes that appear around real frequency peaks when a signal does not contain a whole
number of cycles.

```python
# Taper the edges before saving, to avoid clicks at the start/end
sig.window("hann").to_wav("tapered.wav")

# Window, then plot the tapered waveform
sig.window("blackman").plot(title="Blackman-windowed")
```

| `window` value | Character |
|----------------|-----------|
| `"hann"` (default) | Low sidelobes, good general use |
| `"hamming"` | Slightly higher sidelobes, better frequency resolution |
| `"blackman"` | Very low sidelobes, wider main lobe |
| `"bartlett"` | Triangular taper |

Raises `ValueError` for unrecognised window names.

> **For spectral analysis, use [`.fft(window=...)`](#fftwindownone) instead.** It applies the
> window *and* corrects the magnitude for the window's coherent gain, so a unit-amplitude sine
> still reads a magnitude of 1. `.window()` is a plain taper (`data × window`) with no such
> correction - reach for it when you want the tapered **waveform** itself: removing edge clicks
> before `.to_wav()`, windowed-sinc FIR design, overlap-add framing, or visualising the taper.

---

## Signal - Filters

All filters use a zero-phase Butterworth design (`scipy.signal.sosfiltfilt`) so they
introduce no time delay. The `order` parameter controls how sharp the roll-off is -
higher order = steeper but more prone to ringing.

Cutoff frequencies must be strictly between 0 Hz and the Nyquist frequency
(`sample_rate / 2`). Violating this raises a `ValueError`.

---

### `.lowpass(cutoff, order=4)`

Pass frequencies below `cutoff`, attenuate everything above.

```python
# Remove high-frequency hiss above 4000 Hz
clean = sig.lowpass(cutoff=4000)

# Sharper roll-off
clean = sig.lowpass(cutoff=4000, order=8)
```

---

### `.highpass(cutoff, order=4)`

Pass frequencies above `cutoff`, attenuate everything below.

```python
# Remove low-frequency rumble below 80 Hz
clean = sig.highpass(cutoff=80)
```

---

### `.bandpass(low, high, order=4)`

Pass only frequencies between `low` and `high`. Everything outside is attenuated.

```python
# Telephone bandwidth: 300–3400 Hz
voice = sig.bandpass(300, 3400)

# Isolate a musical instrument's range
violin = sig.bandpass(196, 3136)
```

---


## Signal - Convolution and Correlation

Two fundamental DSP operations for combining and comparing signals. Both accept either
another `Signal` (which must share the same sample rate) or a raw 1-D NumPy array, and both
return a new `Signal`.

---

### `.convolve(other, mode="full")`

Convolve the signal with another signal or a kernel. Convolution is the operation behind
**FIR filtering**: passing a kernel (impulse response) applies that filter to the signal. It
is computed efficiently in the frequency domain via `scipy.signal.fftconvolve`.

```python
import numpy as np

# Smooth a signal with a 5-tap moving-average FIR kernel
kernel = np.ones(5) / 5
smoothed = sig.convolve(kernel, mode="same")

# Apply a recorded impulse response (convolution reverb)
wet = dry.convolve(impulse_response, mode="full")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `other` | `Signal \| np.ndarray` | - | Another signal (same sample rate) or a 1-D kernel array |
| `mode` | `str` | `"full"` | Output length: `"full"`, `"same"` (length of this signal, centred), or `"valid"` (only fully overlapping points) |

Raises `ValueError` if `other` is a `Signal` with a different sample rate, or is not 1-D.

---

### `.correlate(other, mode="full")`

Cross-correlate the signal with another signal or array. Cross-correlation measures how
similar two signals are as one is slid past the other; the **peak of the result indicates the
shift at which they align best**. For the lag itself in seconds, use `.time_delay()`.

```python
# Cross-correlation sequence (useful for plotting)
xcorr = template.correlate(recording)
xcorr.plot(title="Cross-correlation")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `other` | `Signal \| np.ndarray` | - | Another signal (same sample rate) or a 1-D array |
| `mode` | `str` | `"full"` | Output length: `"full"`, `"same"`, or `"valid"` |

Returns a `Signal` holding the cross-correlation values.

---

### `.time_delay(other)`

Estimate the time delay (in seconds) between this signal and `other`, using the location of
the cross-correlation peak. This is the standard technique for **time-delay estimation** - for
example, finding the offset between two microphones.

A **positive** result means this signal is delayed relative to `other` (its features arrive
later); a **negative** result means it arrives earlier.

```python
# Recover the delay between two microphone recordings
delay = mic_a.time_delay(mic_b)
print(f"mic_a lags mic_b by {delay * 1000:.1f} ms")
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `other` | `Signal \| np.ndarray` | Another signal (same sample rate) or a 1-D array |

Returns a `float` - the estimated delay in seconds. Raises `ValueError` if `other` is a
`Signal` with a different sample rate, or is not 1-D.

---

### `.find_peaks(min_height=None, min_distance=None)`

Find the times of local maxima (peaks) in the signal. This is the standard tool for
**event detection** - locating pulses, heartbeats, or vibration spikes in a waveform.

```python
# Heartbeats in an ECG, ignoring noise and ringing
beats = ecg.find_peaks(min_height=0.5, min_distance=0.3)
print(f"Detected {len(beats)} beats")
bpm = 60.0 / np.diff(beats.times).mean()

# times and heights unpack as a pair, too
times, heights = ecg.find_peaks(min_height=0.5)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `min_height` | `float` | Minimum amplitude a sample must reach to count as a peak. Defaults to `None` (no threshold) |
| `min_distance` | `float` | Minimum spacing between peaks, in seconds. When two peaks fall closer, the lower is discarded. Defaults to `None` (no constraint) |

Returns a `PeakResult` with `.times` (seconds) and `.heights` (amplitude at each peak), both
1-D `np.ndarray`s in ascending time order. It unpacks as `times, heights = ...` and supports
`len(result)`. Both arrays are empty if no peaks are found. Raises `ValueError` if
`min_distance` is negative.

---


## Signal - I/O and Visualization

### `.to_wav(path)`

Save the signal as a 16-bit PCM WAV file. Returns `self` so it can appear mid-chain.
Values are clipped to `[-1, 1]` before conversion.

```python
sig.to_wav("output.wav")

# Chain: process then save, then keep working
cleaned = sig.bandpass(300, 3400).normalize().to_wav("cleaned.wav").trim(0.1)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` | Destination file path |

---

### `.to_dataframe(value_column, time_column, time_index)`

Export the signal to a pandas `DataFrame` so it can flow back into a pandas pipeline
(resampling, rolling windows, joining with other data, plotting). By default the result has
one row per sample with an `amplitude` value column and a `time` column (seconds).

```python
# Hand a filtered signal back to pandas
df = sig.bandpass(300, 3000).to_dataframe()
df.rolling(window=10).mean()

# Custom column names
df = sig.to_dataframe(value_column="voltage", time_column="t_s")

# Omit the time column
df = sig.to_dataframe(time_column=None)

# Put the time axis in the index - round-trips through from_pandas with the
# sample rate re-inferred automatically
df = sig.to_dataframe(time_index=True)
Signal.from_pandas(df, column="amplitude")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `value_column` | `str` | `"amplitude"` | Name of the column holding the signal amplitudes |
| `time_column` | `str \| None` | `"time"` | Name of the sample-time column (seconds). `None` omits it. Ignored when `time_index` is True. |
| `time_index` | `bool` | `False` | If True, place the sample times in the DataFrame index instead of a column |

Requires `pandas` (`pip install "convolinear[pandas]"`). Raises `ImportError` if pandas is not
installed.

Unlike `.to_wav()`, this returns the `DataFrame` (not `self`) - it is a terminal operation that
hands the data back to pandas rather than continuing a `Signal` chain.

---

### `.to_numpy(include_time, copy)`

Export the signal's samples as a plain NumPy array - the way out of the `Signal` pipeline back
into NumPy, scikit-learn, PyTorch, or any code that just wants raw samples. By default it returns
the 1D amplitude array.

```python
# Drop a filtered signal into a NumPy / scikit-learn pipeline
features = sig.bandpass(300, 3000).to_numpy()

# Keep the time axis alongside the samples - shape (n, 2)
arr = sig.to_numpy(include_time=True)
times, amplitudes = arr[:, 0], arr[:, 1]

# Round-trips through from_numpy (second column is the amplitudes)
Signal.from_numpy(arr, sample_rate=sig.sample_rate, column=1)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_time` | `bool` | `False` | If True, return a 2D `(n, 2)` array of `[time, amplitude]` columns instead of the 1D amplitude array |
| `copy` | `bool` | `True` | If True, return a fresh, mutable array. Pass False to skip the copy and get a read-only view of the samples. Ignored when `include_time` is True. |

Like `.to_dataframe()`, this returns the array (not `self`) - a terminal operation that hands the
data back to NumPy rather than continuing a `Signal` chain.

---

### `.plot(title, xlabel, ylabel, ax)`

Plot the signal in the time domain using matplotlib. Returns the `Axes` object.

```python
sig.plot()
sig.plot(title="Raw recording")

# Custom axis labels
sig.plot(xlabel="Time (s)", ylabel="Voltage (V)")

# Embed in an existing figure
fig, axes = plt.subplots(2, 1)
sig.plot(ax=axes[0], title="Before")
filtered.plot(ax=axes[1], title="After")
plt.tight_layout()
plt.show()
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `title` | `str \| None` | `"Signal"` | Plot title |
| `xlabel` | `str \| None` | `"Time (s)"` | X-axis label |
| `ylabel` | `str \| None` | `"Amplitude"` | Y-axis label |
| `ax` | `Axes \| None` | `None` | Existing matplotlib `Axes` to draw on. Creates a new figure if `None`. |

---

### `.fft(window=None)`

Convert to the frequency domain. Returns a `Spectrum` object.

```python
spectrum = sig.fft()
print(spectrum.peak_frequency)  # dominant frequency in Hz
```

An optional `window` function reduces spectral leakage - visible as spurious
sidelobes around real frequency peaks when a signal does not contain a whole number
of cycles. Use `"hann"` as a good general-purpose choice.

```python
# Rectangular window (default) - no processing, fastest
spec = sig.fft()

# Hann window - recommended for most audio and sensor analysis
spec = sig.fft(window="hann")
```

| `window` value | Character |
|----------------|-----------|
| `None` (default) | Rectangular - no windowing |
| `"hann"` | Low sidelobes, good general use |
| `"hamming"` | Slightly higher sidelobes, better frequency resolution |
| `"blackman"` | Very low sidelobes, wider main lobe |
| `"bartlett"` | Triangular taper |

Raises `ValueError` for unrecognised window names.

---

### `.spectrogram(segment_length=256, overlap=0.5, window="hann")`

Compute a **spectrogram** via the short-time Fourier transform (STFT). The signal is split
into overlapping segments; each is windowed and Fourier-transformed, producing a picture of
how the frequency content evolves over time. This is the standard tool for non-stationary
signals - speech, music, chirps - whose spectrum changes as they play. Returns a
[`Spectrogram`](#spectrogram--reference) object.

```python
import numpy as np
from convolinear import Signal

# A frequency sweep from 200 Hz to 1000 Hz
chirp = Signal.from_function(
    lambda t: np.sin(2 * np.pi * (200 + 400 * t) * t),
    duration=2.0,
)

chirp.spectrogram().plot(max_freq=2000)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `segment_length` | `int` | `256` | Samples per STFT segment. Larger = finer frequency resolution but coarser time resolution. |
| `overlap` | `float` | `0.5` | Fraction of overlap between consecutive segments, in `[0, 1)`. |
| `window` | `str` | `"hann"` | Window applied to each segment: `"hann"`, `"hamming"`, `"blackman"`, or `"bartlett"`. |

Magnitudes use the same convention as `.fft()`: a sinusoid of amplitude 1 reads a magnitude
of roughly 1 in the bins and frames where it is present.

Raises `ValueError` if `window` is unrecognised or `overlap` is not in `[0, 1)`.

---

## Spectrum - Reference

`Spectrum` objects are produced by `Signal.fft()`. They hold paired arrays of frequencies
(Hz) and magnitudes. All magnitudes are normalised so that a sine wave of amplitude 1
has a magnitude of 1 at its frequency.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `frequencies` | `np.ndarray` | Frequency values in Hz for each bin |
| `magnitudes` | `np.ndarray` | Amplitude at each frequency |
| `peak_frequency` | `float` | The frequency with the highest magnitude |
| `peak_magnitude` | `float` | The magnitude at the peak frequency |
| `len(spec)` | `int` | Number of frequency bins |

```python
spec = Signal.sine(440, duration=1.0, sample_rate=8000).fft()

print(spec.peak_frequency)   # 440.0
print(spec.peak_magnitude)   # ≈ 1.0
print(len(spec))             # 4001
print(spec)
# Spectrum(bins=4001, freq_range=(0.0, 4000.0) Hz)
```

---

### `spec.top_n(n=5)`

Return the `n` largest peaks as a list of `(frequency, magnitude)` tuples, sorted by
magnitude descending.

```python
t = np.arange(8000) / 8000
mixed = np.sin(2 * np.pi * 200 * t) + np.sin(2 * np.pi * 800 * t)
spec = Signal(mixed, sample_rate=8000).fft()

for freq, mag in spec.top_n(2):
    print(f"{freq:.0f} Hz  magnitude={mag:.3f}")
# 200 Hz  magnitude=1.000
# 800 Hz  magnitude=1.000
```

---

### `spec.in_range(low, high)`

Return a new `Spectrum` containing only frequencies in `[low, high]` Hz.

```python
spec = Signal.from_wav("audio.wav").fft()

# Look at just the sub-bass region
sub_bass = spec.in_range(20, 80)
print(sub_bass.peak_frequency)
```

---

### `spec.to_signal(sample_rate)`

Reconstruct a time-domain `Signal` via the inverse FFT. Because a `Spectrum` stores only
magnitudes (not phase), the reconstructed signal has **zero phase** - all components are
cosines. This is useful for synthesis and spectral shaping, but is not a lossless round-trip
from an original recording.

```python
sig = Signal.sine(440, duration=1.0)
spec = sig.fft()

# Keep only the 300–600 Hz band, then synthesise back to time domain
filtered_spec = spec.in_range(300, 600)
reconstructed = filtered_spec.to_signal(sample_rate=44100)
reconstructed.plot(title="Band-limited synthesis")
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `sample_rate` | `int` | Sample rate of the output Signal in Hz |

---

### `spec.plot(title, xlabel, ylabel, log_scale, max_freq, ax)`

Plot the magnitude spectrum. Returns the `Axes` object.

```python
spec.plot()

# Log scale is useful for audio; limit display to 8 kHz
spec.plot(title="Spectrum", log_scale=True, max_freq=8000)

# Custom axis labels
spec.plot(xlabel="Frequency (Hz)", ylabel="Magnitude (linear)")

# Embed in a figure
fig, (ax1, ax2) = plt.subplots(1, 2)
spec.plot(ax=ax1, title="Full spectrum")
spec.in_range(0, 2000).plot(ax=ax2, title="Low frequencies")
plt.show()
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `title` | `str \| None` | `"Frequency Spectrum"` | Plot title |
| `xlabel` | `str \| None` | `"Frequency (Hz)"` | X-axis label |
| `ylabel` | `str \| None` | `"Magnitude"` | Y-axis label |
| `log_scale` | `bool` | `False` | Use logarithmic Y axis for magnitude |
| `max_freq` | `float \| None` | `None` | Limit the X axis to this frequency in Hz |
| `ax` | `Axes \| None` | `None` | Existing `Axes` to draw on |

---

## Spectrogram - Reference

`Spectrogram` objects are produced by `Signal.spectrogram()`. They hold a 2-D array of
magnitudes indexed by frequency (rows) and time (columns), together with the frequency and
time axes that label them.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `frequencies` | `np.ndarray` | Frequency value (Hz) for each row, from 0 to Nyquist |
| `times` | `np.ndarray` | Centre time (seconds) of each STFT frame |
| `magnitudes` | `np.ndarray` | 2-D array of shape `(len(frequencies), len(times))` |
| `shape` | `tuple[int, int]` | Shorthand for `magnitudes.shape` |
| `len(spec)` | `int` | Number of time frames |

```python
spec = Signal.sine(1000, duration=1.0, sample_rate=8000).spectrogram()

print(spec)
# Spectrogram(frequencies=129, frames=..., freq_range=(0.0, 4000.0) Hz, duration=... s)
print(spec.shape)        # (129, ...)
print(spec.frequencies[-1])  # 4000.0  (Nyquist)
```

---

### `spec.peak_frequency_over_time()`

Return the dominant frequency (Hz) in each time frame as an array the same length as `times`.
Useful for tracking how a tone moves over time - a chirp, a glissando, or a vibrato.

```python
chirp = Signal.from_function(
    lambda t: np.sin(2 * np.pi * (200 + 400 * t) * t), duration=2.0
)
track = chirp.spectrogram().peak_frequency_over_time()
print(track[0], "->", track[-1])  # rises from ~200 Hz toward ~1000 Hz
```

---

### `spec.plot(title, xlabel, ylabel, max_freq, db_scale, colorbar, cmap, ax)`

Plot the spectrogram as a heatmap. Returns the matplotlib `Axes`.

```python
spec.plot()

# Limit to 2 kHz, linear magnitude scale
spec.plot(max_freq=2000, db_scale=False)

# Embed alongside the waveform
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
sig.plot(ax=ax1, title="Waveform")
sig.spectrogram().plot(ax=ax2, max_freq=4000)
plt.tight_layout()
plt.show()
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `title` | `str \| None` | `"Spectrogram"` | Plot title |
| `xlabel` | `str \| None` | `"Time (s)"` | X-axis label |
| `ylabel` | `str \| None` | `"Frequency (Hz)"` | Y-axis label |
| `max_freq` | `float \| None` | `None` | Limit the frequency axis to this value in Hz |
| `db_scale` | `bool` | `True` | Show magnitude in decibels. `False` uses a linear scale. |
| `colorbar` | `bool` | `True` | Draw a colour bar alongside the plot |
| `cmap` | `str` | `"magma"` | matplotlib colormap name |
| `ax` | `Axes \| None` | `None` | Existing `Axes` to draw on |

---

## Worked Examples

### 1. Clean a noisy recording

```python
from convolinear import Signal

noisy = Signal.from_wav("field_recording.wav")

cleaned = (
    noisy
    .highpass(80)           # remove low-frequency rumble
    .bandpass(200, 8000)    # keep speech/music range
    .normalize()
)

cleaned.to_wav("cleaned.wav")
print(f"Peak frequency: {cleaned.fft().peak_frequency:.1f} Hz")
```

---

### 2. Mix signals and analyse the result

```python
from convolinear import Signal
import numpy as np

sr = 44100
tone_a = Signal.sine(440, duration=2.0, sample_rate=sr)           # A4
tone_b = Signal.sine(554, duration=2.0, sample_rate=sr)           # C#5
tone_c = Signal.sine(659, duration=2.0, sample_rate=sr)           # E5

chord = tone_a + tone_b + tone_c                                  # mix with +
chord = chord.normalize()
chord.to_wav("chord.wav")

# Confirm all three frequencies appear
for freq, mag in chord.fft().top_n(3):
    print(f"{freq:.0f} Hz  (magnitude {mag:.3f})")
# 440 Hz  (magnitude 0.333)
# 554 Hz  (magnitude 0.333)
# 659 Hz  (magnitude 0.333)
```

---

### 3. Remove mains hum (50 Hz)

```python
from convolinear import Signal

sig = Signal.from_wav("hum_affected.wav")
clean = sig.remove_dc().bandstop(45, 55).normalize()
clean.to_wav("no_hum.wav")
```

---

### 4. Build a test tone with a precise energy level

```python
from convolinear import Signal

target_rms = 0.1
tone = Signal.sine(1000, duration=5.0, sample_rate=44100)

# Scale to exact RMS
scaled = tone.gain(target_rms / tone.rms)
print(f"RMS: {scaled.rms:.4f}")  # 0.1000
scaled.to_wav("reference_tone.wav")
```

---

### 5. Plot before and after filtering

```python
import matplotlib.pyplot as plt
from convolinear import Signal

raw = Signal.from_wav("audio.wav")
filtered = raw.bandpass(300, 3400)

fig, axes = plt.subplots(2, 2, figsize=(14, 6))

raw.trim(0, 0.05).plot(title="Raw (first 50 ms)",      ax=axes[0, 0])
filtered.trim(0, 0.05).plot(title="Filtered (first 50 ms)", ax=axes[0, 1])

raw.fft().plot(title="Raw spectrum",      max_freq=8000, ax=axes[1, 0])
filtered.fft().plot(title="Filtered spectrum", max_freq=8000, ax=axes[1, 1])

plt.tight_layout()
plt.savefig("comparison.png", dpi=120)
```

---

### 4. Load sensor data from a CSV file

```python
from convolinear import Signal

# CSV with columns: timestamp (ISO 8601), voltage
sig = Signal.from_csv(
    "sensor_log.csv",
    value_column="voltage",
    time_column="timestamp",
)

print(sig)                          # Signal(samples=…, sample_rate=… Hz, duration=… s)
print(sig.fft().peak_frequency)     # dominant frequency in the sensor data

sig.highpass(1).normalize().to_wav("sensor.wav")
```

---

### 5. Load a recording saved from MATLAB

```python
from convolinear import Signal

# .mat file with variables: 'ecg' (samples) and 'fs' (sample rate scalar)
sig = Signal.from_matlab(
    "ecg_recording.mat",
    variable="ecg",
    sample_rate_variable="fs",
)

print(sig.duration)
sig.bandpass(0.5, 40).plot(title="ECG - bandpass filtered")
```

---

### 6. Load a FLAC file and inspect its spectrum

```python
from convolinear import Signal

sig = Signal.from_audio("lossless.flac")

spec = sig.fft()
print(f"Peak: {spec.peak_frequency:.1f} Hz @ magnitude {spec.peak_magnitude:.3f}")

spec.plot(title="FLAC spectrum", log_scale=True, max_freq=20000)
```

---

### 7. Visualise a frequency sweep with a spectrogram

```python
import matplotlib.pyplot as plt
import numpy as np
from convolinear import Signal

# A chirp sweeping 200 Hz -> 2000 Hz over 3 seconds
chirp = Signal.from_function(
    lambda t: np.sin(2 * np.pi * (200 + 600 * t) * t),
    duration=3.0,
    sample_rate=16000,
)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
chirp.trim(0, 0.05).plot(ax=ax1, title="Chirp waveform (first 50 ms)")
chirp.spectrogram(segment_length=512).plot(ax=ax2, max_freq=3000)
plt.tight_layout()
plt.savefig("chirp_spectrogram.png", dpi=120)
```

---

### 8. Estimate the delay between two recordings

```python
from convolinear import Signal

# Two microphones picked up the same event a short distance apart
mic_a = Signal.from_wav("mic_a.wav")
mic_b = Signal.from_wav("mic_b.wav")

delay = mic_a.time_delay(mic_b)
print(f"mic_a lags mic_b by {delay * 1000:.2f} ms")

# Distance to the source difference (speed of sound ~343 m/s)
print(f"Path difference: {abs(delay) * 343 * 100:.1f} cm")
```

---

### 9. FIR smoothing by convolution

```python
import numpy as np
from convolinear import Signal

noisy = Signal.from_wav("noisy.wav")

# 11-tap moving-average low-pass, length preserved with mode="same"
kernel = np.ones(11) / 11
smoothed = noisy.convolve(kernel, mode="same").normalize()
smoothed.to_wav("smoothed.wav")
```

---

## Development
If you want to contribute to this project:
1. Fork the repository on GitHub,
2. Run the following git bash commands to set up an editable clone on your local machine:
```bash
git clone https://github.com/yourusername/convolinear
cd convolinear
pip install -e ".[dev]"
pytest
```
3. Make a new branch for your edits,
4. Make changes,
5. Run pytest again to check if anything breaks,
6. Commit changes to your fork,
7. Open a Pull Request.
---

## License

**MIT - © 2026 Sajid Ahmed**