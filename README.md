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
- [Signal - I/O and Visualization](#signal--io-and-visualization)
- [Spectrum - Reference](#spectrum--reference)
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
pip install "convolinear[pandas]"    # pandas + pyarrow - required for from_csv / from_parquet / from_pandas
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
| `pandas` | pandas, pyarrow | `Signal.from_csv()`, `Signal.from_parquet()`, `Signal.from_pandas()` |

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

Load a signal from WAV, FLAC, MP3, OGG, and any other audio format supported by the `soundfile` library.
NOTE: If you are only working with WAV files, using `Signal.from_wav(path)` is recommended as it doesn't require the `soundfile` library.

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

The underlying samples as a `np.ndarray` of dtype `float64`.
NOTE: I recommend treating `sig.data` as read-only.

```python
print(sig.data[:10])
```

### `sig.time_axis`

NumPy array of the time value (in seconds) for each sample. Useful for plotting.

```python
import matplotlib.pyplot as plt
plt.plot(sig.time_axis, sig.data)
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

### `.fft()`

Convert to the frequency domain. Returns a `Spectrum` object.

```python
spectrum = sig.fft()
print(spectrum.peak_frequency)  # dominant frequency in Hz
```

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

### 3. Plot before and after filtering

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
