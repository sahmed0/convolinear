[![PyPI](https://img.shields.io/pypi/v/convolinear.svg)](https://pypi.org/project/convolinear/)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/convolinear.svg)](https://anaconda.org/conda-forge/convolinear)
[![Python](https://img.shields.io/pypi/pyversions/convolinear.svg)](https://pypi.org/project/convolinear/)
[![CI](https://github.com/sahmed0/convolinear/actions/workflows/ci.yml/badge.svg)](https://github.com/sahmed0/convolinear/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-convolinear.sajidahmed.co.uk-blue)](https://convolinear.sajidahmed.co.uk/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/sahmed0/convolinear/blob/main/LICENSE)

# convolinear - Fluent Signal Processing for Python

Fluent signal processing for Python, one line at a time.

<p align="center">
  <img src="https://raw.githubusercontent.com/sahmed0/convolinear/main/docs/assets/hero.png" alt="convolinear pulse compression: the same chirp echo shown at -6 dB SNR, where the sweep is still visible in a spectrogram, and at -20 dB SNR, where it is not; one correlate() call recovers the 1400 ms round-trip delay in both cases" width="900">
</p>

<p align="center">
  <em>Pulse compression - the core of every radar, sonar and GPS receiver. The same echo is shown
  at -6 dB and at -20 dB SNR (echo power against noise power). At -20 dB the spectrogram shows
  nothing at all, and one <code>correlate()</code> call still collapses the echo into a spike at
  the exact round-trip delay. Reproduce with <code>python examples/hero.py</code>.</em>
</p>

`convolinear` wraps the power of NumPy and SciPy in a fluent, readable API. Common DSP tasks -
loading audio, filtering, mixing, and spectral analysis - become short, expressive one-liners
instead of 10+ lines of boilerplate.

Extract data from a `.wav` file, bandpass it, normalise it, take the Fourier transform, and plot it.
**All in a single line of code:**

```python
Signal.from_wav("audio.wav").bandpass(300, 3000).normalize().fft().plot()
```

<p align="center">
  <img src="https://raw.githubusercontent.com/sahmed0/convolinear/main/docs/assets/demo_output.png" alt="convolinear signal processing pipeline result: bandpass filter, normalize, and FFT plot of an audio signal" width="700">
</p>

## Installation

**Core requirements:** Python 3.11+, NumPy >= 1.23.2, SciPy >= 1.9.2

**Optional dependencies** unlock the following loaders and features:

| Extra | Packages installed | Unlocks |
|-------|--------------------|---------|
| `plot` | matplotlib | every `.plot()` method |
| `audio` | soundfile | `Signal.from_audio()` |
| `pandas` | pandas, pyarrow | `from_csv()`, `from_parquet()`, `from_pandas()`, `to_dataframe()` |

MATLAB `.mat` files load through SciPy, already a core dependency - no extra needed.

### With pip

`convolinear` is published on [PyPi](https://pypi.org/project/convolinear):

```bash
pip install convolinear
```

```bash
pip install "convolinear[plot,audio,pandas]"
```

### With conda

`convolinear` is published on [conda-forge](https://anaconda.org/conda-forge/convolinear):

```bash
conda install -c conda-forge convolinear
```

conda-forge packages don't support pip-style extras - install the optional dependencies as
separate packages instead:

```bash
conda install -c conda-forge convolinear matplotlib soundfile pandas pyarrow
```

### With uv

```bash
uv add convolinear
```

```bash
uv add "convolinear[plot,audio,pandas]"
```

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

- **[Full documentation and API reference](https://convolinear.sajidahmed.co.uk/)**
- [Quickstart](https://convolinear.sajidahmed.co.uk/quickstart/) - five worked recipes.
- [ECG: heart rate from raw samples](https://convolinear.sajidahmed.co.uk/examples/ecg_heart_rate/) -
  raw signal to beats per minute.
- [Daily data: finding slow cycles](https://convolinear.sajidahmed.co.uk/examples/daily_cycles/) -
  sub-1 Hz sample rates.
- [Benchmarks](https://convolinear.sajidahmed.co.uk/examples/benchmarks/) - measured throughput.
- [CHANGELOG](https://github.com/sahmed0/convolinear/blob/main/CHANGELOG.md) - release history and migration notes.

Runnable scripts live in [`examples/`](https://github.com/sahmed0/convolinear/tree/main/examples).

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

## License

This project is licensed under the [MIT License](https://github.com/sahmed0/convolinear/blob/main/LICENSE).

Copyright © 2026 Sajid Ahmed
