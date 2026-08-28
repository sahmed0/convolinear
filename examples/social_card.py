"""Social card: a painted spectrogram at 1200x630, on labelled axes.

The chrome is drawn dark so it sits on the same ground as the spectrogram
instead of boxing it into a white page.

1200x630 is the aspect ratio GitHub, X, LinkedIn and Slack all tend to use.

Run with:  python examples/social_card.py [--wordmark]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np

from convolinear import Signal

SR = 44_100
WORDMARK = "--wordmark" in sys.argv
OUT = Path(__file__).parent.parent / ("social_card_wordmark.png" if WORDMARK else "social_card.png")

# Frame the spectrogram above the DC bin: the near-DC row reads as a hard orange
# edge along the bottom of the plot rather than as signal.
FREQ_LO, FREQ_HI = 120, 3200

BG = "#0b0710"  # matches the low end of inferno, so the chrome blends into the plot
INK = "#ece7f2"
MUTED = "#8d8397"


def paint(t: np.ndarray) -> np.ndarray:
    """Steer instantaneous frequency to draw a curve *in* the spectrogram.

    The curve is a damped sine - a ringdown settling towards its rest frequency,
    similar to the amplitude of a damped harmonic oscillator.

    Amplitude and centre are set so the curve stays clear of both frame edges -
    a trough clipped by the bottom of the plot reads as a botched crop, not as
    design.
    """
    envelope = 1200 * np.exp(-t / 3.0)
    freq = 1550 + envelope * np.sin(2 * np.pi * 0.6 * t)
    return np.sin(2 * np.pi * np.cumsum(freq) / SR)


sig = (
    Signal.from_function(paint, duration=6.0, sample_rate=SR)
    .remove_dc()
    .fade_in(0.05)
    .fade_out(0.05)
)

# 1200x630 exactly. The axes and colourbar are placed by hand so the panel keeps
# a predictable size no matter what the tick labels do.
plt.rcParams.update({"font.size": 13, "text.color": INK})
fig = plt.figure(figsize=(12.0, 6.3), dpi=100, facecolor=BG)
ax = fig.add_axes((0.077, 0.115, 0.798, 0.775))
cax = fig.add_axes((0.888, 0.115, 0.016, 0.775))

sig.spectrogram(segment_length=2048, overlap=0.95).plot(
    max_freq=FREQ_HI,
    cmap="inferno",
    colorbar=False,
    title="Signal.from_function(ringdown).spectrogram().plot()",
    ax=ax,
)

# Stretch the colour range across the noise floor so the curve blazes.
mesh = ax.collections[0]
values = np.asarray(mesh.get_array())
mesh.set_clim(np.percentile(values, 40), np.percentile(values, 99.9))

ax.set_ylim(FREQ_LO, FREQ_HI)
ax.set_facecolor(BG)
ax.set_title(ax.get_title(), fontsize=17, fontweight="bold", pad=14)
ax.tick_params(colors=MUTED, labelsize=12)
ax.xaxis.label.set(color=INK, size=14)
ax.yaxis.label.set(color=INK, size=14)
for spine in ax.spines.values():
    spine.set_color("#2e2438")

bar = fig.colorbar(mesh, cax=cax, label="Magnitude (dB)")
bar.outline.set_color("#2e2438")
cax.tick_params(colors=MUTED, labelsize=11)
cax.yaxis.label.set(color=INK, size=13)

if WORDMARK:
    # Sits low-left, where the curve is descending. The dark stroke keeps it
    # legible if a trough or a leakage streak drifts underneath it.
    ax.text(
        0.028,
        0.06,
        "convolinear",
        transform=ax.transAxes,
        fontsize=32,
        fontweight="bold",
        color="white",
        path_effects=[patheffects.withStroke(linewidth=5, foreground=BG)],
    )

fig.savefig(OUT, dpi=100, facecolor=BG)
print(f"wrote {OUT}")
