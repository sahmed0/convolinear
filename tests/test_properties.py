"""Property-based tests (Hypothesis) sealing the core invariants of the library."""

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from convolinear import Signal

settings.register_profile("default", deadline=None)  # numpy-heavy tests; wall-time varies
settings.load_profile("default")

finite = st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False, width=64)
signal_arrays = arrays(np.float64, st.integers(min_value=1, max_value=2048), elements=finite)
rates = st.floats(min_value=1e-5, max_value=1e6, allow_nan=False, allow_infinity=False)


@given(signal_arrays, rates)
def test_fft_roundtrip_is_lossless(data, rate):
    sig = Signal(data, rate)
    rt = sig.fft().to_signal()
    tol = 1e-9 * max(1.0, float(np.max(np.abs(data))))
    np.testing.assert_allclose(rt.data, sig.data, rtol=0, atol=tol)
    assert rt.sample_rate == sig.sample_rate


@given(signal_arrays, rates)
def test_parseval(data, rate):
    sig = Signal(data, rate)
    c = sig.fft().coefficients
    n = len(sig)
    # Full-spectrum energy from the half spectrum: interior bins appear
    # twice (+/- frequency pair), DC once, and for even n Nyquist once.
    e_freq = (
        2.0 * np.sum(np.abs(c) ** 2)
        - np.abs(c[0]) ** 2
        - (np.abs(c[-1]) ** 2 if n % 2 == 0 else 0.0)
    ) / n
    e_time = float(np.sum(data**2))
    np.testing.assert_allclose(e_freq, e_time, rtol=1e-9, atol=1e-6)


@given(
    st.integers(1, 1024).flatmap(
        lambda n: st.tuples(
            arrays(np.float64, n, elements=finite),
            arrays(np.float64, n, elements=finite),
        )
    ),
    finite,
    finite,
    rates,
)
def test_fft_linearity(arrays_pair, a, b, rate):
    x, y = arrays_pair
    combined = Signal(a * x + b * y, rate).fft().coefficients
    expected = a * Signal(x, rate).fft().coefficients + b * Signal(y, rate).fft().coefficients
    scale = max(1.0, float(np.max(np.abs(x))), float(np.max(np.abs(y))))
    np.testing.assert_allclose(combined, expected, rtol=1e-7, atol=1e-6 * scale)


@given(signal_arrays, rates)
def test_reverse_is_an_involution(data, rate):
    sig = Signal(data, rate)
    np.testing.assert_array_equal(sig.reverse().reverse().data, sig.data)


@given(signal_arrays, rates, st.floats(0, 10), st.floats(0, 10))
def test_trim_never_returns_garbage(data, rate, start, end):
    sig = Signal(data, rate)
    n = len(sig)
    try:
        result = sig.trim(start, end)
    except ValueError:
        return
    expected_len = min(int(end * rate), n) - int(start * rate)
    assert len(result) == expected_len
    assert len(result) >= 1


@given(signal_arrays, rates)
def test_normalize_bounds(data, rate):
    assume(np.any(data != 0))
    normalized = Signal(data, rate).normalize()
    assert np.max(np.abs(normalized.data)) == pytest.approx(1.0)
    assert np.all(np.abs(normalized.data) <= 1.0 + 1e-12)


@given(
    st.integers(1, 512).flatmap(
        lambda n: st.tuples(
            arrays(np.float64, n, elements=finite),
            arrays(np.float64, n, elements=finite),
        )
    ),
    rates,
)
def test_convolve_is_commutative(arrays_pair, rate):
    x, y = arrays_pair
    left = Signal(x, rate).convolve(y).data
    right = Signal(y, rate).convolve(x).data
    # fftconvolve's absolute error grows with the output magnitude, so scale
    # atol by it - a fixed atol is far too tight for large inputs, and rtol
    # alone can't cover the near-zero cancellation entries.
    scale = max(1.0, float(np.max(np.abs(left))), float(np.max(np.abs(right))))
    np.testing.assert_allclose(left, right, rtol=1e-7, atol=1e-6 * scale)


@given(signal_arrays, signal_arrays, rates)
def test_concat_length_arithmetic(a_data, b_data, rate):
    a = Signal(a_data, rate)
    b = Signal(b_data, rate)
    joined = a.concat(b)
    assert len(joined) == len(a) + len(b)
    np.testing.assert_array_equal(joined.data[len(a) :], b.data)
