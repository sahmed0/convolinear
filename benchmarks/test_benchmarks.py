"""Benchmarks for convolution and filtering.

These live outside ``testpaths`` (see ``[tool.pytest.ini_options]``), so a plain ``pytest`` run
never collects them. Run them explicitly::

    uv run --group bench pytest benchmarks --benchmark-only --benchmark-sort=name

Results are published in ``docs/examples/benchmarks.md``.
"""

import numpy as np
import pytest

from convolinear import Signal

SAMPLE_RATE = 44100.0


def _noise(n: int) -> Signal:
    """A fixed-seed noise signal of ``n`` samples, so every run measures the same work."""
    rng = np.random.default_rng(0)
    return Signal(rng.standard_normal(n), SAMPLE_RATE)


@pytest.mark.parametrize("n", [1_000, 100_000, 1_000_000])
@pytest.mark.parametrize("kernel_size", [11, 101, 1001])
@pytest.mark.parametrize("method", ["fft", "direct"])
def test_convolve_fft_vs_direct(benchmark, n: int, kernel_size: int, method: str) -> None:
    """Compare Signal.convolve (scipy fftconvolve) against numpy's direct convolution."""
    sig = _noise(n)
    kernel = np.ones(kernel_size) / kernel_size

    if method == "fft":
        benchmark(lambda: sig.convolve(kernel, mode="same"))
    else:
        data = sig.data
        benchmark(lambda: np.convolve(data, kernel, mode="same"))


@pytest.mark.parametrize("n", [10_000, 100_000, 1_000_000])
def test_filter_throughput(benchmark, n: int) -> None:
    """Throughput of a 4th-order Butterworth lowpass at 44.1 kHz."""
    sig = _noise(n)
    benchmark(lambda: sig.lowpass(1000))
