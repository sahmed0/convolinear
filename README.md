[![CI](https://github.com/sahmed0/convolinear/actions/workflows/ci.yml/badge.svg)](https://github.com/sahmed0/convolinear/actions/workflows/ci.yml)
[![MIT License](https://img.shields.io/badge/MIT-2026_Sajid_Ahmed-limegreen.svg)](https://opensource.org/license/mit)
[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://www.python.org/)

# convolinear

A clean, chainable Python library for digital signal processing.

`convolinear` wraps the power of NumPy and SciPy in a fluent, readable API. Common DSP tasks -
loading audio, filtering, mixing, and spectral analysis - become short, expressive one-liners
instead of 10+ lines of boilerplate.

Extract data from a `.wav` file, bandpass it, normalise it, take the Fourier transform, and plot it.
**All in a single line of code:**

```python
Signal.from_wav("audio.wav").bandpass(300, 3000).normalize().fft().plot()
```

<p align="center">
  <img src="https://github.com/sahmed0/convolinear/blob/main/demo_output.png?raw=true" alt="demo.py output graphs" width="700">
</p>

## Installation

```bash
pip install convolinear
```

Optional extras unlock the loaders and features you need:

```bash
pip install "convolinear[plot,audio,pandas]"
```

| Extra | Packages installed | Unlocks |
|-------|--------------------|---------|
| `plot` | matplotlib | every `.plot()` method |
| `audio` | soundfile | `Signal.from_audio()` |
| `pandas` | pandas, pyarrow | `from_csv()`, `from_parquet()`, `from_pandas()`, `to_dataframe()` |

MATLAB `.mat` files load through SciPy, already a core dependency - no extra needed.

**Core requirements:** Python 3.11+, NumPy >= 1.23.2, SciPy >= 1.8

## Quickstart

Clean up a noisy tone, then find out what is in it:

```python
from convolinear import Signal

tone = Signal.sine(440, duration=1.0, sample_rate=8000)
noise = Signal.noise(duration=1.0, sample_rate=8000, amplitude=0.5, seed=42)

cleaned = (tone + noise).bandpass(300, 600).normalize()

spectrum = cleaned.fft()
print(f"Dominant frequency: {spectrum.peak_frequency:.1f} Hz")   # ~440.0 Hz

for freq, magnitude in spectrum.top_n(3):
    print(f"{freq:7.1f} Hz   magnitude {magnitude:.4f}")
```

Every transformation returns a **new** object - the original is never modified, and a `Signal`'s
sample array is genuinely read-only - so branching from one source is always safe.

### Not just audio

Sample rates are floats, and rates below 1 Hz work exactly like any other. Three years of daily
temperatures, one sample per day:

```python
sig = Signal(temps, sample_rate=1 / 86400)
cycle_hz = sig.remove_dc().fft(window="hann").peak_frequency
print(f"Dominant period: {1 / cycle_hz / 86400:.0f} days")   # Dominant period: 365 days
```

## Capabilities

- **Load from anywhere** - WAV, FLAC/OGG/MP3 (via soundfile), CSV, Parquet, NumPy, pandas, MATLAB,
  or generated tones, noise and arbitrary functions. Sample rates are inferred from dated data.
- **Transform** - normalize, trim, gain, concat, resample, window, remove DC, reverse, clip, fade.
- **Filter** - Butterworth lowpass, highpass, bandpass and bandstop.
- **Analyse** - convolution and correlation, time-delay estimation, peak detection.
- **Frequency domain** - `fft()` returns a `Spectrum` of raw complex coefficients that inverts
  losslessly; `psd()` gives a Welch power spectral density; `spectrogram()` shows how content
  evolves over time.
- **Plot** - one-call matplotlib views of any signal, spectrum, PSD or spectrogram.

## Documentation

- **[Full documentation and API reference](https://sahmed0.github.io/convolinear/)**
- [Quickstart](https://sahmed0.github.io/convolinear/quickstart/) - five worked recipes.
- [ECG: heart rate from raw samples](https://sahmed0.github.io/convolinear/examples/ecg_heart_rate/) -
  raw signal to beats per minute.
- [Daily data: finding slow cycles](https://sahmed0.github.io/convolinear/examples/daily_cycles/) -
  sub-1 Hz sample rates.
- [Benchmarks](https://sahmed0.github.io/convolinear/examples/benchmarks/) - measured throughput.
- [CHANGELOG](CHANGELOG.md) - release history and migration notes.
- [LICENSE](LICENSE) - MIT.

Runnable scripts live in [`examples/`](examples/).

## Development

If you want to contribute to this project:

1. Fork the repository on GitHub.
2. Set up an editable clone with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/yourusername/convolinear
cd convolinear
uv sync
uv run pytest
```

3. Make a new branch for your edits.
4. Make changes.
5. Run the same checks CI runs:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy convolinear
uv run pytest
```

6. Commit changes to your fork.
7. Open a Pull Request.
