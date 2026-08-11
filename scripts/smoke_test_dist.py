"""Pre-release smoke test: exercise the *built* artifacts as an external user would.

Unlike `pytest`, this never runs against the editable source tree - each case
runs in a throwaway env containing only the wheel/sdist plus the extras named
in that case, via `uv run --isolated --no-project`.

CI runs this on Linux, macOS, and Windows: the wheel is py3-none-any, but the
dependency wheels are not, and `soundfile` bundles a native library.

Usage:
    uv run scripts/smoke_test_dist.py
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"

# Several times the slowest realistic call (a cold pandas+pyarrow install), so
# tripping it means hung, not slow. Keep comfortably under the workflow timeout.
SUBPROCESS_TIMEOUT = 300

# The package must stay small: pure Python source and nothing else.
MAX_WHEEL_BYTES = 200_000
MAX_SDIST_BYTES = 200_000

# Everything the wheel is allowed to contain, beyond `convolinear/*.py`.
WHEEL_EXTRA_ALLOWED = {"convolinear/py.typed"}

# Everything the sdist is allowed to contain, beyond `convolinear/*.py`.
SDIST_EXTRA_ALLOWED = {
    "convolinear/py.typed",
    "README.md",
    "LICENSE",
    "PKG-INFO",
    "pyproject.toml",
    ".gitignore",
}

# Must never appear anywhere in a built artifact.
FORBIDDEN_DIR_NAMES = {
    "tests",
    "test",
    "benchmarks",
    "bench",
    "docs",
    "doc",
    "examples",
    "scripts",
    "site",
    ".github",
}

# Every extra declared in pyproject; asserts the wheel metadata still
# advertises them under the names users type.
EXPECTED_EXTRAS = {"plot", "pandas", "audio"}


# A declared requirement this script knows how to pin: a bare name and a
# single `>=` floor. Anything richer has no unambiguous "the floor" to pin.
FLOOR_RE = re.compile(r"^(?P<name>[A-Za-z0-9._-]+)\s*>=\s*(?P<version>[0-9][A-Za-z0-9.*+!-]*)$")


def _floor_pins() -> list[str]:
    """Turn every declared `>=` floor in pyproject into an exact `==` pin.

    Extras and the transitive tree are deliberately left unpinned - pinning
    them would only ever exercise combinations no real user runs.
    """
    with (ROOT / "pyproject.toml").open("rb") as fh:
        declared: list[str] = list(tomllib.load(fh)["project"]["dependencies"])

    pins: list[str] = []
    for req in declared:
        match = FLOOR_RE.match(req.strip())
        if match is None:
            sys.exit(
                f"pyproject declares {req!r}, which is not a plain `name>=version` "
                "floor; the floor case cannot tell which version to pin."
            )
        pins.append(f"{match['name']}=={match['version']}")
    return pins


BASE_FLOOR_PINS = _floor_pins()

# --- Case snippets -----------------------------------------------------
# Each snippet runs in its own isolated environment with only the given
# extras installed and must reach the final marker print to pass.

MARKER = "SMOKE OK"

# Prepended to snippets that check packaging metadata.
PRELUDE = f"EXPECTED_EXTRAS = set({sorted(EXPECTED_EXTRAS)!r})\n"

BASE = (
    PRELUDE
    + """
import numpy as np
import convolinear
from convolinear import Signal

# The wheel must be what we imported - not a stray source tree on sys.path.
assert "site-packages" in convolinear.__file__, convolinear.__file__

from importlib.metadata import metadata, version
assert convolinear.__version__ == version("convolinear"), (
    f"__version__ {convolinear.__version__} != dist metadata {version('convolinear')}"
)

for name in convolinear.__all__:
    assert getattr(convolinear, name, None) is not None, f"__all__ names missing {name}"

declared = set(metadata("convolinear").get_all("Provides-Extra") or [])
assert declared == EXPECTED_EXTRAS, f"extras {declared} != expected {EXPECTED_EXTRAS}"

# None of these may be present, or the fail-closed assertions below would
# pass vacuously.
for mod in ("matplotlib", "pandas", "soundfile"):
    try:
        __import__(mod)
    except ImportError:
        pass
    else:
        raise AssertionError(f"{mod} leaked into the base environment")

# Core round trip: no optional dependency should be required for this.
sig = Signal(np.sin(2 * np.pi * 5 * np.linspace(0, 1, 1000, endpoint=False)), sample_rate=1000)
spec = sig.fft()
back = spec.to_signal()
assert np.allclose(sig.data, back.data, atol=1e-9), "fft -> to_signal round trip is lossy"

assert sig.psd().peak_frequency > 0
assert sig.spectrogram().shape[0] > 0
assert sig.lowpass(100).rms > 0

# WAV I/O is core (scipy), not gated behind the 'audio' extra.
import tempfile, os
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "out.wav")
    sig.to_wav(path)
    reloaded = Signal.from_wav(path)
    assert len(reloaded) == len(sig)

# Each optional path must fail closed with a clear ImportError, not an
# AttributeError or a confusing traceback.
cases = [
    ("Signal.plot", lambda: sig.plot()),
    ("Spectrum.plot", lambda: sig.fft().plot()),
    ("PowerSpectrum.plot", lambda: sig.psd().plot()),
    ("Spectrogram.plot", lambda: sig.spectrogram().plot()),
    ("Signal.to_dataframe", lambda: sig.to_dataframe()),
    ("Signal.from_pandas", lambda: Signal.from_pandas(object())),
    ("Signal.from_csv", lambda: Signal.from_csv("nonexistent.csv", value_column="v")),
    ("Signal.from_parquet", lambda: Signal.from_parquet("nonexistent.parquet", value_column="v")),
    ("Signal.from_audio", lambda: Signal.from_audio("nonexistent.mp3")),
]
for name, call in cases:
    try:
        call()
    except ImportError:
        pass
    else:
        raise AssertionError(f"{name} should have raised ImportError without its extra")

print("SMOKE OK")
"""
)

PLOT = """
import matplotlib
matplotlib.use("Agg")
import numpy as np
from convolinear import Signal

sig = Signal(np.sin(np.linspace(0, 10, 500)), sample_rate=500)

assert sig.plot() is not None
assert sig.fft().plot() is not None
assert sig.psd().plot() is not None
assert sig.spectrogram().plot() is not None

# The plot extra must not have dragged in the others.
for mod in ("pandas", "soundfile"):
    try:
        __import__(mod)
    except ImportError:
        pass
    else:
        raise AssertionError(f"{mod} leaked into the plot-only environment")

print("SMOKE OK")
"""

PANDAS = """
import os, tempfile
import numpy as np
import pandas as pd
from convolinear import Signal

sig = Signal(np.linspace(-1, 1, 200), sample_rate=200)
df = sig.to_dataframe(time_index=True)
assert isinstance(df, pd.DataFrame)
back = Signal.from_pandas(df, column="amplitude")
assert len(back) == len(sig)

with tempfile.TemporaryDirectory() as d:
    csv_path = os.path.join(d, "s.csv")
    sig.to_dataframe().to_csv(csv_path, index=False)
    from_csv = Signal.from_csv(csv_path, value_column="amplitude", time_column="time")
    assert len(from_csv) == len(sig)

    # pyarrow is declared in this extra solely so from_parquet works; it must be exercised
    # otherwise there is no way to tell whether the declaration is needed.
    pq_path = os.path.join(d, "s.parquet")
    sig.to_dataframe().to_parquet(pq_path, index=False)
    from_pq = Signal.from_parquet(pq_path, value_column="amplitude", time_column="time")
    assert len(from_pq) == len(sig)

print("SMOKE OK")
"""

AUDIO = """
import tempfile, os
import numpy as np
from convolinear import Signal

sig = Signal(np.sin(np.linspace(0, 20, 4000)), sample_rate=4000)
with tempfile.TemporaryDirectory() as d:
    import soundfile as sf

    # FLAC, not WAV: WAV is already covered without this extra (via scipy, in
    # BASE). FLAC exercises the bundled native library this extra exists for.
    path = os.path.join(d, "out.flac")
    sf.write(path, sig.to_numpy(), int(sig.sample_rate))
    back = Signal.from_audio(path)
    assert len(back) == len(sig)
print("SMOKE OK")
"""

# Proves py.typed is not just present in the wheel but actually makes the
# package's types visible to an external consumer.
TYPING_CONSUMER = """
import numpy as np
from convolinear import Signal


def peak(sig: Signal) -> float:
    return sig.fft().peak_frequency


s = Signal(np.zeros(128), sample_rate=128)
reveal_type(peak(s))
"""


@dataclass
class Case:
    label: str
    snippet: str
    extras: str = ""
    python: str | None = None
    # Exact `==` pins installed alongside the artifact (see `_floor_pins`).
    pins: list[str] = field(default_factory=list)
    # Install from the sdist rather than the wheel.
    from_sdist: bool = False


CASES: list[Case] = [
    Case("base (no extras)", BASE),
    Case("base @ python 3.14 (highest classified version)", BASE, python="3.14"),
    # Holds core deps at their declared pyproject floors, on the minimum
    # Python (3.11) - also covers the declared Python floor.
    Case(
        "base @ declared dependency floors",
        BASE,
        python="3.11",
        pins=BASE_FLOOR_PINS,
    ),
    Case("base from sdist", BASE, from_sdist=True),
    Case("plot extra", PLOT, extras="[plot]"),
    Case("pandas extra", PANDAS, extras="[pandas]"),
    Case("audio extra", AUDIO, extras="[audio]"),
]

# Every target `pip install convolinear[<extra>]` must resolve. Checked via a
# dry-run resolution rather than a full download of matplotlib + pandas +
# pyarrow + soundfile.
PIP_EXTRA_TARGETS = ["", "[plot]", "[pandas]", "[audio]"]

# None of these may appear in a bare `pip install convolinear`.
OPTIONAL_PACKAGES = {"matplotlib", "pandas", "pyarrow", "soundfile"}


def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    """`subprocess.run`, but a hang becomes a failed result instead of an
    exception that would abandon the other checks queued behind it."""
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("text", True)
    kwargs.setdefault("timeout", SUBPROCESS_TIMEOUT)
    try:
        return subprocess.run(cmd, **kwargs)
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            cmd,
            1,
            (exc.stdout or "") if isinstance(exc.stdout, str) else "",
            f"timed out after {kwargs['timeout']}s: {' '.join(cmd)}",
        )


def find_artifacts() -> tuple[Path, Path]:
    wheels = sorted(DIST.glob("convolinear-*.whl"))
    sdists = sorted(DIST.glob("convolinear-*.tar.gz"))
    if not wheels or not sdists:
        sys.exit(f"No wheel/sdist found in {DIST}. Run `uv build` first.")
    # `main` wipes dist/ before building, so anything else means a stale
    # artifact would silently be the one under test.
    if len(wheels) != 1 or len(sdists) != 1:
        sys.exit(
            f"Expected exactly one wheel and one sdist in {DIST}, "
            f"found {len(wheels)}/{len(sdists)}."
        )
    return wheels[0], sdists[0]


def check_artifacts(wheel: Path, sdist: Path) -> bool:
    """Assert the built artifacts contain the source and nothing else."""
    problems: list[str] = []

    def classify(names: list[str], allowed: set[str], kind: str) -> None:
        for name in names:
            if name.endswith("/"):
                continue
            parts = name.split("/")
            if "__pycache__" in parts or name.endswith((".pyc", ".pyo")):
                problems.append(f"{kind}: compiled artifact {name}")
            elif set(parts) & FORBIDDEN_DIR_NAMES:
                problems.append(f"{kind}: non-source directory in payload: {name}")
            elif parts[0] == "convolinear" and name.endswith(".py"):
                continue
            elif name in allowed:
                continue
            else:
                problems.append(f"{kind}: unexpected file {name}")

    with zipfile.ZipFile(wheel) as zf:
        wheel_names = zf.namelist()
    # dist-info is generated by the backend, not my problem. Match on the
    # directory suffix so a stray `convolinear-extra-data/` isn't exempted.
    classify(
        [n for n in wheel_names if not n.split("/", 1)[0].endswith(".dist-info")],
        WHEEL_EXTRA_ALLOWED,
        "wheel",
    )
    if "convolinear/py.typed" not in wheel_names:
        problems.append("wheel: py.typed is missing - the package ships untyped")

    with tarfile.open(sdist) as tf:
        # Strip the leading `convolinear-<version>/` directory component.
        sdist_names = [n.split("/", 1)[1] for n in tf.getnames() if "/" in n]
    classify(sdist_names, SDIST_EXTRA_ALLOWED, "sdist")
    if "convolinear/py.typed" not in sdist_names:
        problems.append("sdist: py.typed is missing")

    for path, ceiling in ((wheel, MAX_WHEEL_BYTES), (sdist, MAX_SDIST_BYTES)):
        size = path.stat().st_size
        if size > ceiling:
            problems.append(f"{path.name} is {size} bytes, over the {ceiling} ceiling")

    ok = not problems
    print(f"[{'PASS' if ok else 'FAIL'}] artifact contents")
    for p in problems:
        print(f"    {p}")
    return ok


def check_metadata(wheel: Path, sdist: Path) -> tuple[str, bool, str]:
    """Run `twine check` - the metadata validation PyPI applies at upload."""
    result = run(
        [
            "uv",
            "run",
            "--isolated",
            "--no-project",
            "--with",
            "twine",
            "twine",
            "check",
            str(wheel),
            str(sdist),
        ]
    )
    # twine hard-wraps its output, so match the verdict token rather than a phrase.
    ok = result.returncode == 0 and "FAILED" not in result.stdout.upper()
    label = "twine check (PyPI metadata + README rendering)"
    return label, ok, "" if ok else result.stdout + result.stderr


def check_typing(wheel: Path) -> tuple[str, bool, str]:
    """Type-check a consumer module against the installed wheel."""
    with tempfile.TemporaryDirectory() as tmp:
        consumer = Path(tmp) / "consumer.py"
        consumer.write_text(TYPING_CONSUMER, encoding="utf-8")
        result = run(
            [
                "uv",
                "run",
                "--isolated",
                "--no-project",
                "--with",
                str(wheel),
                "--with",
                "mypy",
                "mypy",
                "--strict",
                str(consumer),
            ],
            cwd=tmp,
        )
    # mypy renders this as either `float` or `builtins.float` by version.
    revealed = next((ln for ln in result.stdout.splitlines() if "Revealed type is" in ln), "")
    # Match the exact phrase for "py.typed is missing or unreadable", not the
    # bare word "stub", to avoid a false failure on an unrelated output containing "stub".
    ok = (
        revealed.endswith(('"float"', '"builtins.float"'))
        and "Cannot find implementation or library stub" not in result.stdout
    )
    label = "py.typed usable by downstream mypy"
    return label, ok, "" if ok else result.stdout + result.stderr


def _venv_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def check_pip_extras(wheel: Path) -> tuple[str, bool, str]:
    """Assert pip can resolve every install target, from wheels alone.

    Runs pip rather than uv - different resolver, and what most users
    actually type. `--only-binary :all:` additionally asserts every runtime
    dependency has a wheel on this platform.
    """
    label = "pip resolves every install target (wheels only)"
    problems: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        venv = Path(tmp) / "venv"
        created = run([sys.executable, "-m", "venv", str(venv)])
        if created.returncode != 0:
            return label, False, created.stderr
        python = _venv_python(venv)

        for extras in PIP_EXTRA_TARGETS:
            report = Path(tmp) / f"report{extras or '-base'}.json"
            result = run(
                [
                    str(python),
                    "-m",
                    "pip",
                    "install",
                    "--dry-run",
                    "--quiet",
                    "--only-binary",
                    ":all:",
                    "--report",
                    str(report),
                    f"{wheel}{extras}",
                ]
            )
            if result.returncode != 0 or not report.exists():
                problems.append(
                    f"pip could not resolve 'convolinear{extras}': {result.stderr.strip()}"
                )
                continue
            data = json.loads(report.read_text(encoding="utf-8"))
            resolved = {i["metadata"]["name"].lower().replace("_", "-") for i in data["install"]}

            for core in ("numpy", "scipy"):
                if core not in resolved:
                    problems.append(f"convolinear{extras} did not pull in core dep {core}")
            if not extras:
                leaked = OPTIONAL_PACKAGES & resolved
                if leaked:
                    problems.append(f"bare install pulls in optional packages {sorted(leaked)}")

    ok = not problems
    return label, ok, "" if ok else "\n".join(problems)


def run_case(case: Case, wheel: Path, sdist: Path) -> tuple[str, bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        script_path = Path(tmp) / "case.py"
        script_path.write_text(case.snippet, encoding="utf-8")
        artifact = sdist if case.from_sdist else wheel
        cmd = ["uv", "run", "--isolated", "--no-project"]
        if case.python:
            cmd += ["--python", case.python]
        for pin in case.pins:
            cmd += ["--with", pin]
        cmd += ["--with", f"{artifact}{case.extras}", "python", str(script_path)]
        result = run(cmd, cwd=tmp)

    # Exit code 0 is not enough: the snippet must have run to the end.
    ok = result.returncode == 0 and MARKER in result.stdout
    detail = ""
    if not ok:
        if result.returncode == 0:
            detail = f"case exited 0 but never reached the '{MARKER}' marker\n"
        detail += result.stdout + result.stderr
    return case.label, ok, detail


def main() -> int:
    if shutil.which("uv") is None:
        sys.exit("uv is required on PATH to run this script.")

    # CI pipes stdout, which block-buffers it - without this a run killed by
    # the job timeout shows nothing, not even the checks that passed.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)

    print("Building sdist + wheel...")
    if DIST.exists():
        shutil.rmtree(DIST)
    built = run(["uv", "build"], cwd=ROOT)
    if built.returncode != 0:
        sys.exit(f"uv build failed:\n{built.stdout}{built.stderr}")

    wheel, sdist = find_artifacts()
    print(f"Testing {wheel.name} in isolated environments (dev deps excluded)...\n")

    results: list[bool] = [check_artifacts(wheel, sdist)]

    # Each case builds its own environment, so they're independent. Submit
    # `check_pip_extras` first since it's the slowest (stdlib venv, no uv cache).
    workers = min(len(CASES) + 3, max(4, (os.cpu_count() or 4) * 2))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(check_pip_extras, wheel)]
        futures += [pool.submit(run_case, c, wheel, sdist) for c in CASES]
        futures.append(pool.submit(check_metadata, wheel, sdist))
        futures.append(pool.submit(check_typing, wheel))

        # Report in completion order: if the job is killed, finished checks
        # have already been printed and aren't lost.
        for future in as_completed(futures):
            label, ok, detail = future.result()
            print(f"[{'PASS' if ok else 'FAIL'}] {label}")
            if detail:
                print(detail)
            results.append(ok)

    print()
    if all(results):
        print(f"All {len(results)} smoke-test checks passed.")
        return 0
    failed = sum(1 for r in results if not r)
    print(f"{failed}/{len(results)} smoke-test checks FAILED.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
