"""convolinear - A fluent time-series and signal processing library for building data pipelines in a single line of Python."""

from .signal import Signal, PeakResult
from .spectrum import Spectrum
from .spectrogram import Spectrogram

__version__ = "0.2.0"
__all__ = ["Signal", "PeakResult", "Spectrum", "Spectrogram"]
