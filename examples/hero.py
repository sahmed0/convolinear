"""Hero image: pulse compression, the workflow behind radar, sonar and GPS.

Transmit a known wideband chirp, listen to a noisy return, then cross-correlate
the return against a clean copy of what was sent. The correlation collapses the
smeared-out echo into a single spike whose position is the round-trip delay -
even when the echo is far too weak to see in a spectrogram.

The two panels on top are the same noise with the same seed; only the echo
amplitude differs. At -6 dB SNR the sweep is still visible because a
time-frequency cell concentrates the chirp's energy. At -20 dB it is not
visible at all, and correlation still recovers the delay exactly.

SNR here is a power ratio: a sine of amplitude ``a`` has mean square
``a**2 / 2``, so against unit-variance noise the SNR is
``10 * log10(a**2 / 2)`` - that is -6.0 dB at a = 0.7 and -20.0 dB at
a = 0.1414.

Run with:  python examples/hero.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from convolinear import Signal

SR = 8000
OUT = Path(__file__).parent.parent / "docs" / "assets" / "hero.png"
TRUE_DELAY = 1.40  # seconds of round-trip delay we are trying to recover
LOUD_AMPLITUDE = 0.7  # -6 dB SNR against unit-variance noise
QUIET_AMPLITUDE = 0.1414  # -20 dB SNR against unit-variance noise
NOISE_SIGMA = 1.0

# --- Build the data ---------------------------------------------------------

# Reference waveform: a 1.2 s linear FM sweep, 300 Hz -> 3300 Hz.
ref = Signal.from_function(
    lambda t: np.sin(2 * np.pi * np.cumsum(300 + 2500 * t) / SR),
    duration=1.2,
    sample_rate=SR,
)


def receive(amplitude: float) -> Signal:
    """Return the echo at ``amplitude``, delayed by TRUE_DELAY and buried in noise."""
    echo = np.concatenate(
        [np.zeros(round(TRUE_DELAY * SR)), amplitude * ref.data, np.zeros(round(0.9 * SR))]
    )
    noise = Signal.noise(len(echo) / SR, SR, amplitude=NOISE_SIGMA, seed=1).data
    return Signal(echo + noise, SR)


def snr_db(amplitude: float) -> float:
    """Echo power relative to noise power, in dB."""
    return float(10 * np.log10(np.mean((amplitude * ref.data) ** 2) / NOISE_SIGMA**2))


loud = receive(LOUD_AMPLITUDE)
quiet = receive(QUIET_AMPLITUDE)

# --- Process the data --------------------------------------------------------

compressed = Signal(quiet.correlate(ref).data[len(ref) - 1 :], SR).normalize()
estimate = quiet.time_delay(ref)

# --- Render ------------------------------------------------------------------

plt.rcParams.update({"font.size": 11, "axes.titlesize": 11.5, "axes.titleweight": "bold"})
fig = plt.figure(figsize=(12, 6.4), layout="constrained")
grid = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.05, wspace=0.04)
ax_loud = fig.add_subplot(grid[0, 0])
ax_quiet = fig.add_subplot(grid[0, 1], sharey=ax_loud)
ax_corr = fig.add_subplot(grid[1, :])

for sig, ax, caption in (
    (loud, ax_loud, f"{snr_db(LOUD_AMPLITUDE):.0f} dB SNR  -  the sweep still shows"),
    (quiet, ax_quiet, f"{snr_db(QUIET_AMPLITUDE):.0f} dB SNR  -  nothing to see"),
):
    sig.spectrogram(segment_length=256, overlap=0.85).plot(
        max_freq=3800, cmap="magma", colorbar=False, title=caption, ax=ax
    )

# Stretch the colour range across the noise floor, and use the same range for
# both panels so the eye is comparing like with like.
quiet_values = np.asarray(ax_quiet.collections[0].get_array())
clim = (np.percentile(quiet_values, 2), np.percentile(quiet_values, 99.95))
for ax in (ax_loud, ax_quiet):
    ax.collections[0].set_clim(*clim)
ax_quiet.set_ylabel("")
ax_quiet.tick_params(labelleft=False)

compressed.plot(ax=ax_corr, ylabel="Correlation")
ax_corr.set_title(
    "received.correlate(ref)  -  pulse compression locks the arrival time at "
    f"{snr_db(QUIET_AMPLITUDE):.0f} dB SNR"
)
ax_corr.lines[0].set(color="#1b3a6b", linewidth=0.7)
ax_corr.set_xlim(*ax_loud.get_xlim())
ax_corr.set_ylim(-0.75, 1.35)

ax_corr.axvline(estimate, color="#e03030", lw=1.2, ls="--", zorder=1)
ax_corr.plot([estimate], [1.0], marker="o", ms=9, mfc="none", mec="#e03030", mew=2, zorder=3)
ax_corr.annotate(
    f"round trip = {estimate * 1000:.0f} ms",
    xy=(estimate, 1.0),
    xytext=(estimate + 0.22, 1.12),
    color="#e03030",
    fontweight="bold",
    arrowprops={"arrowstyle": "-", "color": "#e03030", "lw": 1.2},
)

fig.suptitle("convolinear  -  fluent signal processing for Python", fontsize=15, fontweight="bold")
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=160)
print(f"wrote {OUT}")
print(
    f"true delay {TRUE_DELAY * 1000:.0f} ms, estimated {estimate * 1000:.0f} ms; "
    f"SNR {snr_db(LOUD_AMPLITUDE):.1f} dB and {snr_db(QUIET_AMPLITUDE):.1f} dB"
)
