# Quickstart

Five short recipes covering the paths most people need first. Each one stands alone.

## 1. Clean up a recording and save it

Filters chain, and every step returns a new `Signal`.

```python
from convolinear import Signal

noisy = Signal.from_wav("field_recording.wav")

cleaned = (
    noisy
    .highpass(80)         # drop low-frequency rumble
    .bandpass(200, 8000)  # keep the speech/music range
    .normalize()          # scale the peak to 1.0
)

cleaned.to_wav("cleaned.wav")
print(f"Peak frequency: {cleaned.fft().peak_frequency:.1f} Hz")
```

`to_wav()` requires an integer sample rate and will tell you to `.resample(...)` if the signal's
rate is fractional.

## 2. Generate tones, mix them, find the peaks

`+` mixes two signals of equal length. `top_n` returns true spectral peaks - local maxima found
with `scipy.signal.find_peaks` - not just the largest bins.

```python
from convolinear import Signal

sr = 44100
chord = (
    Signal.sine(440, duration=2.0, sample_rate=sr)   # A4
    + Signal.sine(554, duration=2.0, sample_rate=sr)  # C#5
    + Signal.sine(659, duration=2.0, sample_rate=sr)  # E5
).normalize()

for freq, mag in chord.fft().top_n(3):
    print(f"{freq:.0f} Hz  (magnitude {mag:.3f})")
# 659 Hz  (magnitude 0.334)
# 554 Hz  (magnitude 0.334)
# 440 Hz  (magnitude 0.334)
```

Peaks come back ordered by descending magnitude, not by frequency. `top_n` returns *up to* `n` of
them - fewer if the spectrum has fewer qualifying peaks. Pass `min_prominence=` to suppress noise.

## 3. Filter in the frequency domain and come back

`Signal.fft()` stores the raw complex coefficients, so a `Spectrum` inverts exactly.
`in_range` zeroes everything outside the band and keeps the axis full length, which makes it an
ideal brick-wall filter:

```python
import numpy as np
from convolinear import Signal

two_tone = Signal.sine(100, 1.0, 4000) + Signal.sine(1200, 1.0, 4000)

# Lossless round-trip: the FFT throws nothing away
np.testing.assert_allclose(two_tone.fft().to_signal().data, two_tone.data, atol=1e-12)

# Keep only the low tone, then return to the time domain
low_only = two_tone.fft().in_range(0, 500).to_signal()
print(f"{low_only.fft().peak_frequency:.0f} Hz")  # 100 Hz
```

To synthesise a signal *from* a magnitude spectrum instead, use
`Spectrum.from_magnitudes(mags, freqs, sample_rate).to_signal()`.

## 4. See how frequency content changes over time

```python
import numpy as np
from convolinear import Signal

# A chirp sweeping 200 Hz -> 2000 Hz over 3 seconds
chirp = Signal.from_function(
    lambda t: np.sin(2 * np.pi * (200 + 600 * t) * t),
    duration=3.0,
    sample_rate=16000,
)

chirp.spectrogram(segment_length=512).plot(max_freq=3000, title="Chirp")
```

For a *statistical* view of noisy data, reach for `psd()` instead - Welch's method averages
overlapping segments, so unlike a single FFT its variance shrinks as the signal gets longer:

```python
noisy = Signal.sine(50, 10.0, 1000) + Signal.noise(10.0, 1000, amplitude=2.0, seed=0)
print(f"{noisy.psd(segment_length=1024).peak_frequency:.0f} Hz")  # 50 Hz, despite the noise
```

Longer segments buy frequency resolution at the cost of fewer averages: at `sample_rate=1000`, the
default 256-sample segment resolves only to ~3.9 Hz, while 1024 narrows that to ~1 Hz.

## 5. Round-trip through pandas

Sample rates are inferred from a `DatetimeIndex` or a time column - including rates below 1 Hz.

```python
import pandas as pd
from convolinear import Signal

df = pd.read_csv("sensor_log.csv", parse_dates=["timestamp"])
sig = Signal.from_pandas(df.set_index("timestamp")["voltage"])

print(sig)  # Signal(samples=..., sample_rate=... Hz, duration=... s)

back = sig.remove_dc().to_dataframe(value_column="voltage", time_index=True)
```

`Signal.from_csv("sensor_log.csv", value_column="voltage", time_column="timestamp")` does the same
in one step.
