# Benchmarks

`Signal.convolve` computes convolution in the frequency domain via `scipy.signal.fftconvolve`.
This page measures what that actually buys, against NumPy's direct time-domain `np.convolve`.

**Machine:** 12th Gen Intel Core i5-12500H (12 physical / 16 logical cores), 16 GB RAM,
Windows 11 (25H2), Python 3.14.5, NumPy 2.4.6, SciPy 1.17.1, pytest-benchmark 5.2.3.

Reproduce with:

```bash
uv run --group bench pytest benchmarks --benchmark-only --benchmark-sort=name
```

## Convolution: FFT vs direct

Minimum time over all rounds - the most stable estimator, and the least polluted by scheduling
noise. Input is fixed-seed Gaussian noise; the kernel is a moving average of the given length.

| Signal length | Kernel taps | `Signal.convolve` (FFT) | `np.convolve` (direct) | Faster |
|--------------:|------------:|------------------------:|-----------------------:|:-------|
| 1,000 | 11 | 0.049 ms | **0.008 ms** | direct, 5.8x |
| 1,000 | 101 | 0.052 ms | **0.015 ms** | direct, 3.6x |
| 1,000 | 1,001 | 0.061 ms | **0.058 ms** | direct, 1.1x |
| 100,000 | 11 | 3.11 ms | **0.21 ms** | direct, 14.8x |
| 100,000 | 101 | 4.07 ms | **1.37 ms** | direct, 3.0x |
| 100,000 | 1,001 | **3.70 ms** | 7.27 ms | **FFT, 2.0x** |
| 1,000,000 | 11 | 47.8 ms | **4.7 ms** | direct, 10.3x |
| 1,000,000 | 101 | 53.2 ms | **18.0 ms** | direct, 3.0x |
| 1,000,000 | 1,001 | **53.9 ms** | 84.2 ms | **FFT, 1.6x** |

## Where the crossover is

The two rows that matter are the last of each block. **An FFT convolution's cost is set almost
entirely by the length of the signal, not the kernel:** at one million samples it takes 47.8, 53.2
and 53.9 ms for kernels of 11, 101 and 1,001 taps - essentially flat across a 90x change in kernel
size, because the transform is the same size either way. **Direct convolution instead costs roughly
`n x k`**, so it starts far cheaper and climbs linearly with every tap you add; the two cross at
somewhere around 300-1,000 taps, largely independent of `n`.

The practical reading: the frequency-domain approach is the right default for long FIR kernels,
where it wins by 1.6-2.0x and keeps winning as the kernel grows. For short kernels - the 11-tap
moving average of the smoothing recipes - it is 3-15x *slower* than the direct method, because
`Signal.convolve` pays for a full forward and inverse transform of the whole signal to do work that
a handful of multiply-adds per sample would have finished sooner. `Signal.convolve` always takes the
FFT path, so if you are convolving with a very short kernel in a hot loop, calling `np.convolve` on
`sig.data` directly is measurably faster.

## Filter throughput

A 4th-order Butterworth lowpass at 1 kHz cutoff, 44.1 kHz sample rate:

| Signal length | Time (min) | Throughput |
|--------------:|-----------:|-----------:|
| 10,000 | 0.52 ms | ~19 Msamples/s |
| 100,000 | 1.89 ms | ~53 Msamples/s |
| 1,000,000 | 15.5 ms | ~64 Msamples/s |

Throughput improves with length as the fixed per-call overhead (coefficient design, array
allocation) amortises, settling around 64 Msamples/s - roughly 24 minutes of 44.1 kHz audio filtered
per second of compute.
