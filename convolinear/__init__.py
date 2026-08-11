"""convolinear - Fluent signal processing for Python, one line at a time."""

from .power_spectrum import PowerSpectrum
from .signal import PeakResult, Signal
from .spectrogram import Spectrogram
from .spectrum import Spectrum

__version__ = "0.2.0"
__all__ = ["PeakResult", "PowerSpectrum", "Signal", "Spectrogram", "Spectrum"]
