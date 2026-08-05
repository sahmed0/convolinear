# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

A correctness and feature release. The frequency-domain classes were rebuilt so that a transform can be
undone, several silent-wrong-answer bugs were fixed, and every claim the package makes — its typing
marker, its Python support matrix — is now backed by something that runs in CI. There are breaking
changes and no compatibility shims; the notes below are the migration path.

### Added

- `PowerSpectrum` and `Signal.psd()` — power spectral density estimation with Welch's method, for noisy
  data where a single-shot FFT has variance that never shrinks.
- `Spectrum.coefficients`, `.phase` and `.power` — the raw complex coefficients and derived views over
  them. Phase information is now available at all.
- `Spectrum.from_magnitudes()` — zero-phase synthesis from a magnitude spectrum. This is where the old
  lossy `to_signal(sample_rate)` behaviour went; spectral-shaping workflows use this now.
- Float sample rates, including rates below 1 Hz. Daily-sampled data loads at its true `1/86400 Hz`
  instead of being rounded to zero and rejected.
- `min_prominence` on `Spectrum.top_n()`, for suppressing noise peaks.
- A property-based test suite (Hypothesis) sealing the round-trip, Parseval, linearity, involution and
  commutativity invariants across randomised inputs.
- Continuous integration: lint, format, type-check and the full suite on Python 3.11–3.14 across Linux,
  Windows and macOS, with a coverage floor.

### Changed

- **`Spectrum` stores complex coefficients instead of magnitudes.** `to_signal()` takes no argument and
  is now a lossless inverse — `sig.fft().to_signal()` returns the original signal to floating-point
  precision. Magnitude values are unchanged; `.magnitudes` reads exactly as before.
- **The `Spectrum` constructor** now takes `(coefficients, frequencies, n_samples, sample_rate, scale)`.
  For synthesis from magnitudes, use `Spectrum.from_magnitudes(mags, freqs, sample_rate).to_signal()`.
- **`Spectrum.in_range()` zeroes out-of-band bins rather than dropping them.** The result keeps the full
  frequency axis and stays invertible, so `in_range(...).to_signal()` is a brick-wall filter. It raises
  if the band contains no bins.
- **`Spectrum.top_n()` returns true spectral peaks**, found with `scipy.signal.find_peaks`, rather than
  the `n` largest bins — the largest bins of a leaky spectrum are usually the same lobe sampled several
  times. It returns *up to* `n` peaks, possibly fewer or none.
- **`Signal.data` and `Signal.sample_rate` are read-only properties** over frozen arrays; the same holds
  for the arrays on `Spectrum`, `Spectrogram` and `PowerSpectrum`. Writing to them raises. Arrays passed
  into a constructor are copied first, so the caller's own array is never frozen.
- **Empty signals and spectra are rejected at construction**, so every object that exists supports every
  method. `trim()` raises rather than returning an empty signal, and `convolve(mode="valid")` raises when
  the kernel is longer than the signal.
- Sample rates are floats throughout. `Signal.to_wav()` raises with guidance if the rate is not
  integer-valued rather than writing a file with a wrong header.
- The package is `mypy --strict` clean, so the shipped `py.typed` marker is now accurate.

### Fixed

- 8-bit WAV files decoded as if they were two's-complement, leaving every sample in `[0, 1]` with a large
  DC offset. They are unsigned offset-binary and now decode to `[-1, 1)` centred on zero.
- Signed PCM was normalised by `iinfo.max` (32767 for int16), pushing full-scale negative samples past
  -1.0. It now divides by `|iinfo.min|` (32768), mapping the full integer range into `[-1, 1)`.
- `trim()` silently returned garbage or an empty signal for a negative start, an end before the start, or
  a range past the end of the signal. It now validates its arguments and caps an over-long end.
- Degenerate spectra and spectrograms could be constructed and then crash on use; they are now rejected
  at construction.
- The duplicated dev-dependency stanza in `pyproject.toml`, which let the two copies drift apart.

## [0.1.0] - 2026-05-22

- Initial release: the `Signal`, `Spectrum` and `Spectrogram` classes, with a fluent API covering
  generation, WAV/CSV/Parquet/NumPy/MATLAB/pandas loading, filtering, mixing, convolution, correlation,
  peak finding, spectral analysis and plotting.
