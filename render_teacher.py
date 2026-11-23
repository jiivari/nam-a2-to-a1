#!/usr/bin/env python3
"""Stage 1/2: run a training signal through a NAM A2 model (the teacher).

Runs in the *new* NAM environment (.venv, neural-amp-modeler 0.13+), which can
read A2 files (SlimmableContainer, model version 0.7). The full-size submodel is
used. Output is an .npz file consumed by a2_to_a1.py.

Usage:
    .venv/bin/python render_teacher.py MODEL_A2.nam OUT.npz --input SIGNAL.wav
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
GAINS = [0.5, 1.0, 1.6]  # teach the student the amp's response at several input levels


def load_teacher(path):
    d = json.load(open(path))
    if d.get("architecture") == "SlimmableContainer":
        sub = max(d["config"]["submodels"], key=lambda s: s["max_value"])["model"]
    elif d.get("architecture") == "WaveNet":
        sub = d
    else:
        raise SystemExit(f"Unsupported architecture: {d.get('architecture')}")
    return init_from_nam(sub).eval(), d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--input", required=True, help="training signal (.wav, any rate)")
    ap.add_argument("--val-seconds", type=float, default=10.0)
    a = ap.parse_args()

    teacher, d = load_teacher(a.model)
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    teacher = teacher.to(device)

    x, sr = sf.read(a.input, dtype="float32", always_2d=True)
    x = x[:, 0]
    if sr != SR:
        g = math.gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
    nval = int(a.val_seconds * SR)
    if len(x) < 3 * nval:
        raise SystemExit("Training signal is too short (need at least 30 s).")
    x_tr, x_val = x[:-nval], x[-nval:]

    @torch.no_grad()
    def run(sig):
        return teacher(torch.from_numpy(sig).to(device), pad_start=True).cpu().numpy()

    out = {"x_val": x_val, "y_val": run(x_val).astype(np.float32), "gains": np.array(GAINS)}
    for k, g in enumerate(GAINS):
        xs = (x_tr * g).astype(np.float32)
        out[f"x{k}"] = xs
        out[f"y{k}"] = run(xs).astype(np.float32)
    meta = {k: v for k, v in (d.get("metadata") or {}).items() if k != "date"}
    out["meta_json"] = np.array(json.dumps(meta))
    np.savez(a.out, **out)
    print(f"teacher rendered: {len(x)/SR:.1f}s of signal -> {a.out}")


if __name__ == "__main__":
    main()
