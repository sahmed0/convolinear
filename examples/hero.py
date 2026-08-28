"""Hero image: pulse compression, the workflow behind radar, sonar and GPS.

Transmit a known wideband chirp, listen to a noisy return, then cross-correlate
the return against a clean copy of what was sent. The correlation collapses the
smeared-out echo into a single spike whose position is the round-trip delay -
even when the echo sits below the noise floor.

Run with:  python examples/hero.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from convolinear import Signal

SR = 8000
OUT = Path(__file__).parent.parent / "hero.png"
TRUE_DELAY = 1.40  # seconds of round-trip delay we are trying to recover

# --- Build the data ---------------------------------------------------------

# Reference waveform: a 1.2 s linear FM sweep, 300 Hz -> 3300 Hz.
ref = Signal.from_function(
    lambda t: np.sin(2 * np.pi * np.cumsum(300 + 2500 * t) / SR),
    duration=1.2,
    sample_rate=SR,
)

# Received signal: the echo returns at 1.40 s at 0.7x amplitude, buried in white
# noise about 3 dB louder than the echo itself.
echo = np.concatenate([np.zeros(round(TRUE_DELAY * SR)), 0.7 * ref.data, np.zeros(round(0.9 * SR))])
received = Signal(echo + Signal.noise(len(echo) / SR, SR, amplitude=1.0, seed=1).data, SR)

# --- Process the data --------------------------------------------------------

compressed = Signal(received.correlate(ref).data[len(ref) - 1 :], SR).normalize()
estimate = received.time_delay(ref)

# --- Render ------------------------------------------------------------------

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold"})
fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={"height_ratios": [1.15, 1]})

received.spectrogram(segment_length=256, overlap=0.85).plot(
    max_freq=3800,
    cmap="magma",
    colorbar=False,
    title="received.spectrogram()  -  a chirp echo, 3 dB under the noise floor",
    ax=top,
)
# Stretch the colour range across the noise floor so the sweep stands out.
mesh = top.collections[0]
values = np.asarray(mesh.get_array())
mesh.set_clim(np.percentile(values, 2), np.percentile(values, 99.95))

compressed.plot(ax=bottom, ylabel="Correlation")
bottom.set_title("received.correlate(ref)  -  pulse compression locks the arrival time")
bottom.lines[0].set(color="#1b3a6b", linewidth=0.7)
bottom.set_xlim(*top.get_xlim())
bottom.set_ylim(-0.55, 1.35)

bottom.axvline(estimate, color="#e03030", lw=1.2, ls="--", zorder=1)
bottom.plot([estimate], [1.0], marker="o", ms=9, mfc="none", mec="#e03030", mew=2, zorder=3)
bottom.annotate(
    f"round trip = {estimate * 1000:.0f} ms",
    xy=(estimate, 1.0),
    xytext=(estimate + 0.22, 1.12),
    color="#e03030",
    fontweight="bold",
    arrowprops={"arrowstyle": "-", "color": "#e03030", "lw": 1.2},
)

fig.suptitle("convolinear  -  fluent signal processing for Python", fontsize=15, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(OUT, dpi=160)
print(f"wrote {OUT}  (true delay {TRUE_DELAY * 1000:.0f} ms, estimated {estimate * 1000:.0f} ms)")
