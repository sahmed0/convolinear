"""convolinear - A fluent time-series and signal processing library.

Build data pipelines in a single line of Python.
"""

from .signal import PeakResult, Signal
from .spectrogram import Spectrogram
from .spectrum import Spectrum

__version__ = "0.2.0"
__all__ = ["PeakResult", "Signal", "Spectrogram", "Spectrum"]
