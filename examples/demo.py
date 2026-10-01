"""Demo: clean up a noisy sine wave and analyse it.

Run with: python examples/demo.py
"""

from pathlib import Path

import matplotlib.pyplot as plt

from convolinear import Signal


def main():
    # 1. Generate a clean 440 Hz tone (musical A4) plus high-frequency noise
    clean_tone = Signal.sine(frequency=440, duration=1.0, sample_rate=8000)
    noise = Signal.noise(duration=1.0, sample_rate=8000, amplitude=0.5, seed=42)
    noisy = Signal(clean_tone.data + noise.data, sample_rate=8000)

    print(f"Noisy signal:    {noisy}")

    # 2. Clean it up: bandpass around 440 Hz, then normalise
    cleaned = noisy.bandpass(low=300, high=600).normalize()
    print(f"Cleaned signal:  {cleaned}")

    # 3. Analyse: take the FFT and find the dominant frequency
    spectrum = cleaned.fft()
    print(f"\nDominant frequency: {spectrum.peak_frequency:.1f} Hz (expected ~440 Hz)")
    print("Top 3 peaks:")
    for freq, mag in spectrum.top_n(3):
        print(f"  {freq:7.1f} Hz   magnitude {mag:.4f}")

    # 4. Plot everything
    _fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    noisy.trim(0, 0.05).plot(title="Noisy signal (first 50 ms)", ax=axes[0])
    cleaned.trim(0, 0.05).plot(title="After bandpass filter (first 50 ms)", ax=axes[1])
    spectrum.plot(title="Spectrum of cleaned signal", max_freq=2000, ax=axes[2])
    plt.tight_layout()
    out = Path(__file__).parent.parent / "docs" / "assets" / "demo_output.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=100)
    print(f"\nPlot saved to {out}")


if __name__ == "__main__":
    main()
