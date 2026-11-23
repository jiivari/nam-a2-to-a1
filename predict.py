#!/usr/bin/env python3
"""Predict how well NAM A2 captures will convert to A1, before training.

Every A2 file holds a small "Lite" and a large "Full" submodel trained on the same
amp. If even the Lite version stays close to Full, the capture is simple enough to
fit a small network, and A1-standard (which is close to Lite in size) will usually
learn it well. If Lite already differs a lot, the capture needs the A2-Full
capacity and the conversion tends to stall at a poor result.

Runs in the *new* NAM environment (.venv). Takes a few seconds per file.

Usage:
    .venv/bin/python predict.py FILE.nam [MORE.nam ...] [--input SIGNAL.wav]
"""
import argparse
import json
import math
import os

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

import render_teacher as rt  # tkinter stub, synthetic(), SR
from nam.models._from_nam import init_from_nam

DEFAULT_INPUT = os.path.expanduser(
    "~/Library/Application Support/com.hotone.mp380/wavs/nam_input_wav.wav")

# Calibrated on a handful of captures (Lite-vs-Full ESR -> conversion verdict):
#   0.019-0.044 -> GOOD/OK at 6000 steps;  0.35-0.51 -> POOR even at 20000 steps.
EASY = 0.06
HARD = 0.15


def signal(path):
    parts = []
    if path:
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        x = x[:, 0]
        if sr != rt.SR:
            g = math.gcd(int(sr), rt.SR)
            x = resample_poly(x, rt.SR // g, int(sr) // g).astype(np.float32)
        parts.append(x)
    parts.append(rt.synthetic(np.random.default_rng(7), [0.05, 0.15, 0.35]))
    return torch.from_numpy(np.concatenate(parts))


def predict(path, x):
    d = json.load(open(path))
    if d.get("architecture") != "SlimmableContainer":
        if str(d.get("version", "")).startswith("0.7"):
            return None, "unknown - no Lite/Full pair in this file; try --steps 1500"
        return None, "already A1 - no conversion needed"
    subs = sorted(d["config"]["submodels"], key=lambda s: s["max_value"])
    with torch.no_grad():
        yl = init_from_nam(subs[0]["model"]).eval()(x, pad_start=True).numpy()
        yf = init_from_nam(subs[-1]["model"]).eval()(x, pad_start=True).numpy()
    e = float(np.sum((yl - yf) ** 2) / (np.sum(yf**2) + 1e-12))
    if e <= EASY:
        verdict = "LIKELY GOOD/OK - should convert well with the default steps"
    elif e <= HARD:
        verdict = "UNCERTAIN - try a quick check with --steps 1500 first"
    else:
        verdict = "LIKELY POOR - probably too complex for A1; look for an A1 version"
    return e, verdict


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--input", help="training signal (default: Ampero Editor's, if present)")
    a = ap.parse_args()
    inp = a.input or (DEFAULT_INPUT if os.path.isfile(DEFAULT_INPUT) else None)
    x = signal(inp)
    print(f"Prediction signal: {os.path.basename(inp) + ' + ' if inp else ''}synthetic")
    print(f"(Lite-vs-Full ESR: <= {EASY} likely good, > {HARD} likely poor)")
    for f in a.files:
        try:
            e, verdict = predict(f, x)
        except Exception as ex:  # noqa: BLE001
            e, verdict = None, f"ERROR: {ex}"
        score = f"{e:7.4f}" if e is not None else "      -"
        print(f"{score}  {verdict}\n         {os.path.basename(f)}")


if __name__ == "__main__":
    main()
