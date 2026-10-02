"""Smoke tests for the plotting methods (matplotlib Agg backend via conftest)."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from convolinear import Signal


class TestSignalPlot:
    def test_returns_axes(self):
        ax = Signal.sine(440, 0.1, 8000).plot()
        assert isinstance(ax, Axes)

    def test_title_and_xlabel_applied(self):
        ax = Signal.sine(440, 0.1, 8000).plot(title="My signal", xlabel="seconds")
        assert ax.get_title() == "My signal"
        assert ax.get_xlabel() == "seconds"

    def test_draws_on_supplied_axes(self):
        _, ax = plt.subplots()
        returned = Signal.sine(440, 0.1, 8000).plot(ax=ax)
        assert returned is ax


class TestSpectrumPlot:
    def test_returns_axes(self):
        ax = Signal.sine(440, 0.1, 8000).fft().plot(max_freq=1000, log_scale=True)
        assert isinstance(ax, Axes)

    def test_draws_on_supplied_axes(self):
        _, ax = plt.subplots()
        returned = Signal.sine(440, 0.1, 8000).fft().plot(ax=ax)
        assert returned is ax


class TestSpectrogramPlot:
    def test_returns_axes_db_scale(self):
        ax = Signal.sine(440, 0.5, 8000).spectrogram().plot(db_scale=True)
        assert isinstance(ax, Axes)

    def test_returns_axes_linear_scale(self):
        ax = Signal.sine(440, 0.5, 8000).spectrogram().plot(db_scale=False)
        assert isinstance(ax, Axes)

    def test_draws_on_supplied_axes(self):
        _, ax = plt.subplots()
        returned = Signal.sine(440, 0.5, 8000).spectrogram().plot(ax=ax)
        assert returned is ax

    def test_max_freq_masks_frequency_rows_not_time_columns(self):
        """max_freq selects rows (axis 0); the time axis must survive intact.

        A Spectrogram's values are 2-D (frequency x time), so masking the wrong
        axis would trim time instead - or raise, when the two axes differ in
        length. Comparing against the expected slice also pins the direction of
        the mask, which a shape-only assertion would not.
        """
        spec = Signal.sine(100.0, duration=1.0, sample_rate=1000.0).spectrogram(segment_length=64)
        max_freq = 200.0

        ax = spec.plot(max_freq=max_freq, db_scale=False)

        plotted = np.asarray(ax.collections[0].get_array())
        expected = spec.magnitudes[spec.frequencies <= max_freq]
        assert 0 < expected.shape[0] < spec.shape[0]  # the mask really trims
        assert plotted.shape == expected.shape
        assert plotted.shape[1] == len(spec.times)
        np.testing.assert_allclose(plotted, expected)

    def test_without_max_freq_every_bin_is_plotted(self):
        """The max_freq=None path must pass both arrays through untouched."""
        spec = Signal.sine(100.0, duration=1.0, sample_rate=1000.0).spectrogram(segment_length=64)

        plotted = np.asarray(spec.plot(db_scale=False).collections[0].get_array())

        assert plotted.shape == spec.shape
        np.testing.assert_allclose(plotted, spec.magnitudes)


class TestPowerSpectrumPlot:
    def test_returns_axes(self):
        ax = Signal.sine(50.0, 4.0, 1000.0).psd().plot()
        assert isinstance(ax, Axes)

    def test_max_freq_limits_both_axes_together(self):
        """The frequency axis and the power values must be trimmed in step."""
        psd = Signal.sine(50.0, duration=4.0, sample_rate=1000.0).psd(segment_length=256)
        max_freq = 100.0

        ax = psd.plot(max_freq=max_freq, log_scale=False)

        keep = psd.frequencies <= max_freq
        assert 0 < keep.sum() < len(psd)  # the mask really trims
        line = ax.lines[0]
        np.testing.assert_allclose(line.get_xdata(), psd.frequencies[keep])
        np.testing.assert_allclose(line.get_ydata(), psd.power[keep])
