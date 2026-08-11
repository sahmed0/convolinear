# Daily data: finding slow cycles

Signal processing tools usually assume audio-rate data, and quietly break below 1 Hz - a rate of
"one sample per day" rounds to zero and takes the analysis with it. `convolinear` stores sample
rates as plain floats with no rounding anywhere, so a daily series is just a signal whose rate
happens to be `1/86400` Hz.

That matters because the interesting cycles in daily data - weekly trading rhythms, annual seasonality,
multi-year business cycles - live at frequencies far below 1 Hz, and they respond to exactly the same
FFT, filtering and PSD machinery as a 44.1 kHz recording.

## Finding an annual cycle

Three years of synthetic daily temperatures: an annual cycle buried in noise.

```python
import numpy as np
from convolinear import Signal

days = np.arange(3 * 365)
temps = 12 + 8 * np.sin(2 * np.pi * days / 365.25) + np.random.default_rng(0).normal(0, 2, days.size)

sig = Signal(temps, sample_rate=1 / 86400)      # one sample per day
cycle_hz = sig.remove_dc().fft(window="hann").peak_frequency
print(f"Dominant period: {1 / cycle_hz / 86400:.0f} days")   # Dominant period: 365 days
```

Three things are doing the work here:

- **`sample_rate=1 / 86400`** - about `1.157e-05` Hz. `repr(sig)` reports
  `Signal(samples=1095, sample_rate=1.15741e-05 Hz, duration=94608000.000 s)`; the duration is in
  seconds, as always, so divide by 86400 to read it back in days.
- **`remove_dc()`** - the mean temperature (~12 degrees) would otherwise dominate the spectrum as a
  huge bin at 0 Hz and win the `peak_frequency` argmax outright.
- **`window="hann"`** - three years is not a whole number of 365.25-day cycles, so the tapering
  suppresses the spectral leakage that discontinuity would otherwise smear across neighbouring bins.

The peak lands at `3.17e-08` Hz. Inverting that gives a period of 31.56 million seconds, which is
365 days.

## The same analysis with a PSD

For a noisier series, Welch's method trades frequency resolution for a stabler estimate. Because
the estimate averages overlapping segments, each segment must still be long enough to contain the
cycle you are hunting - a 365-sample segment spans exactly one year:

```python
ps = sig.remove_dc().psd(segment_length=365)
print(f"Dominant period: {1 / ps.peak_frequency / 86400:.0f} days")   # Dominant period: 365 days
ps.plot(max_freq=1e-6, title="Daily temperatures: power spectral density")
```

`PowerSpectrum.plot` uses a log y-axis by default, which is usually what you want - PSDs routinely
span several decades.

## Reading real dated data

Loading a dated series infers the rate for you, with no rounding:

```python
sig = Signal.from_csv("prices.csv", value_column="close", time_column="date")
print(sig.sample_rate)   # ~1.1574074074074073e-05 for daily rows
```

`Signal.from_pandas(df)` does the same for a `DatetimeIndex`. Note the inferred rate is the raw
`1 / median_interval` and generally won't be an exact round number, so compare it with
`pytest.approx` rather than `==`. One consequence worth knowing: `to_wav()` rejects fractional
rates outright, since WAV headers cannot represent them.
