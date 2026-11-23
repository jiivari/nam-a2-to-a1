#!/usr/bin/env python3
"""Stage 1/2: run training signals through a NAM A2 model (the teacher).

Runs in the *new* NAM environment (.venv, neural-amp-modeler 0.13+), which can
read A2 files (SlimmableContainer, model version 0.7). The full-size submodel is
used. Output is an .npz file consumed by a2_to_a1.py.

Training material = the given signal (whole, at several input levels) plus
synthetic pink noise and sine sweeps, so the student sees the amp's response
across the full frequency and level range. Two validation sets are stored:
  - "target":  the given signal at unity gain (what a host such as the Ampero
               Editor will actually play through the converted model)
  - "holdout": freshly generated noise/sweeps never used for training

Usage:
    .venv/bin/python render_teacher.py MODEL_A2.nam OUT.npz [--input SIGNAL.wav]
"""
import argparse
import json
import math
import sys
import types

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

# neural-amp-modeler imports tkinter for its GUI; Homebrew Python often lacks it.
try:
    import tkinter  # noqa: F401
except ImportError:
    class _Dummy:
        def __init__(self, *a, **k):
            pass

    def _any(name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Dummy

    _tk = types.ModuleType("tkinter")
    _tk.filedialog = types.ModuleType("tkinter.filedialog")
    _tk.messagebox = types.ModuleType("tkinter.messagebox")
    for _m in (_tk, _tk.filedialog, _tk.messagebox):
        _m.__getattr__ = _any
    sys.modules.update({"tkinter": _tk, "tkinter.filedialog": _tk.filedialog,
                        "tkinter.messagebox": _tk.messagebox})

from nam.models._from_nam import init_from_nam  # noqa: E402

SR = 48000
GAINS = [0.5, 1.0, 1.6]  # input levels for the given signal


def load_teacher(path):
    d = json.load(open(path))
    if d.get("architecture") == "SlimmableContainer":
        sub = max(d["config"]["submodels"], key=lambda s: s["max_value"])["model"]
    elif d.get("architecture") == "WaveNet":
        sub = d
    else:
        raise SystemExit(f"Unsupported architecture: {d.get('architecture')}")
    return init_from_nam(sub).eval(), d


def pink_noise(n, rng):
    """Approximate pink (1/f) noise, unit RMS."""
    w = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    f[0] = f[1]
    w /= np.sqrt(f)
    w[f < 20] = 0
    p = np.fft.irfft(w, n)
    return p / np.sqrt(np.mean(p**2))


def log_sweep(seconds, f0=20.0, f1=20000.0):
    t = np.arange(int(seconds * SR)) / SR
    k = math.log(f1 / f0)
    return np.sin(2 * np.pi * f0 * seconds / k * (np.exp(t / seconds * k) - 1))


def synthetic(rng, levels, noise_s=4.0, sweep_s=4.0):
    """Pink-noise bursts and sweeps at the given RMS levels, separated by short gaps."""
    parts = []
    gap = np.zeros(int(0.25 * SR))
    for lv in levels:
        parts += [pink_noise(int(noise_s * SR), rng) * lv, gap,
                  log_sweep(sweep_s) * lv * math.sqrt(2), gap]
    return np.concatenate(parts).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--input", help="training signal (.wav, any rate); if omitted, "
                    "a synthetic signal is used (less accurate for real playing)")
    a = ap.parse_args()

    teacher, d = load_teacher(a.model)
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    teacher = teacher.to(device)

    if a.input:
        x, sr = sf.read(a.input, dtype="float32", always_2d=True)
        x = x[:, 0]
        if sr != SR:
            g = math.gcd(int(sr), SR)
            x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
        if len(x) < 10 * SR:
            raise SystemExit("Training signal is too short (need at least 10 s).")
    else:
        print("No training signal given: using a synthetic signal only. For best "
              "results pass --input (e.g. NAM's standard input.wav).")
        x = synthetic(np.random.default_rng(555),
                      levels=[0.03, 0.08, 0.15, 0.25, 0.4], noise_s=6.0, sweep_s=6.0)

    @torch.no_grad()
    def run(sig):
        return teacher(torch.from_numpy(sig).to(device), pad_start=True).cpu().numpy().astype(np.float32)

    rng = np.random.default_rng(1234)
    train = [(x * g).astype(np.float32) for g in GAINS]
    train.append(synthetic(rng, levels=[0.02, 0.06, 0.15, 0.35]))
    holdout = synthetic(np.random.default_rng(98765), levels=[0.04, 0.12, 0.25])

    out = {"n_train": np.array(len(train))}
    for k, s in enumerate(train):
        out[f"x{k}"] = s
        out[f"y{k}"] = run(s)
    out["x_target"], out["y_target"] = x, run(x)
    out["x_holdout"], out["y_holdout"] = holdout, run(holdout)
    meta = {k: v for k, v in (d.get("metadata") or {}).items() if k != "date"}
    out["meta_json"] = np.array(json.dumps(meta))
    np.savez(a.out, **out)
    total = sum(len(s) for s in train) / SR
    print(f"teacher rendered: {len(x)/SR:.1f}s signal, {total:.0f}s training material -> {a.out}")


if __name__ == "__main__":
    main()
