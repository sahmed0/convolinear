"""convolinear - A fluent time-series and signal processing library.

Build data pipelines in a single line of Python.
"""

from .power_spectrum import PowerSpectrum
from .signal import PeakResult, Signal
from .spectrogram import Spectrogram
from .spectrum import Spectrum

__version__ = "0.2.0"
__all__ = ["PeakResult", "PowerSpectrum", "Signal", "Spectrogram", "Spectrum"]
