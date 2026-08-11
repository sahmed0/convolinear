"""Smoke tests for the plotting methods (matplotlib Agg backend via conftest)."""

import matplotlib.pyplot as plt
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
