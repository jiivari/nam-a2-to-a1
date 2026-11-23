#!/usr/bin/env python3
"""Stage 2/2: train an A1-standard WaveNet to reproduce the A2 teacher's output.

Runs in the *old* NAM environment (.venv-a1, neural-amp-modeler 0.11), which
reads and writes A1 files (model version 0.5.4) - the format older hosts such as
the Hotone Ampero (1st gen) editor accept.

By default the student starts from random weights. Optionally pass
--template SOME_A1_STANDARD.nam to warm-start from an existing A1 capture.

Usage:
    .venv-a1/bin/python a2_to_a1.py TEACHER.npz OUT.nam [--template A1.nam] [--steps N]
"""
import argparse
import copy
import json
import sys
import time
import types

import numpy as np
import torch

# neural-amp-modeler imports tkinter for its GUI; Homebrew Python often lacks it.
# Training does not need it, so provide a permissive stub.
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

from nam.models.wavenet import WaveNet  # noqa: E402

MODEL_VERSION = "0.5.4"
SAMPLE_RATE = 48000
# Quality thresholds for the final verdict (error-to-signal ratio, lower is better).
GOOD_ESR = 0.02
OK_ESR = 0.05

# NAM A1 "standard" WaveNet architecture.
_DILATIONS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
A1_STANDARD_CONFIG = {
    "layers": [
        {"input_size": 1, "condition_size": 1, "head_size": 8, "channels": 16,
         "kernel_size": 3, "dilations": _DILATIONS, "activation": "Tanh",
         "gated": False, "head_bias": False},
        {"input_size": 16, "condition_size": 1, "head_size": 1, "channels": 8,
         "kernel_size": 3, "dilations": _DILATIONS, "activation": "Tanh",
         "gated": False, "head_bias": True},
    ],
    "head": None,
    "head_scale": 0.02,
}


def build_a1(config, weights=None, sample_rate=SAMPLE_RATE):
    m = WaveNet(layers_configs=config["layers"], head_config=config["head"],
                head_scale=config["head_scale"], sample_rate=sample_rate)
    if weights is not None:
        m.import_weights(weights)
    return m


def esr(pred, target):
    return float(np.sum((pred - target) ** 2) / (np.sum(target**2) + 1e-12))


@torch.no_grad()
def render(model, x, device):
    was = model.training
    model.eval()
    y = model(torch.from_numpy(x).to(device), pad_start=True).cpu().numpy()
    model.train(was)
    return y


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("teacher_npz")
    ap.add_argument("out")
    ap.add_argument("--template", help="A1-standard .nam to warm-start from")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--ny", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    z = np.load(a.teacher_npz)
    a2_meta = json.loads(str(z["meta_json"]))
    if "n_train" not in z:
        raise SystemExit("Teacher file is from an older version; re-run render_teacher.py")
    n = int(z["n_train"])
    tx = [torch.from_numpy(z[f"x{k}"]) for k in range(n)]
    ty = [torch.from_numpy(z[f"y{k}"]) for k in range(n)]
    vals = {"target": (z["x_target"], z["y_target"]),
            "holdout": (z["x_holdout"], z["y_holdout"])}

    def evaluate(model, dev):
        return {k: esr(render(model, xv, dev), yv) for k, (xv, yv) in vals.items()}

    if a.template:
        tmpl = json.load(open(a.template))
        config = tmpl["config"]
        student = build_a1(config, tmpl["weights"])
        w0 = np.asarray(student._export_weights(), dtype=np.float64)
        assert len(w0) == len(tmpl["weights"]), "template is not a compatible A1 WaveNet"
        assert np.allclose(w0, tmpl["weights"], atol=1e-6), "weight order mismatch"
        init = "template"
    else:
        config = copy.deepcopy(A1_STANDARD_CONFIG)
        student = build_a1(config)
        init = "random"
    rf = student.receptive_field
    student = student.to(device)
    print(f"device={device} init={init} receptive_field={rf}", flush=True)

    # Sample training windows in proportion to segment length.
    L = rf - 1 + a.ny
    lens = np.array([len(t) - L for t in tx], dtype=np.float64)
    if (lens <= 0).any():
        raise SystemExit("A training segment is shorter than one training window.")
    probs = lens / lens.sum()

    student.train()
    opt = torch.optim.Adam(student.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.1 ** (1.0 / a.steps))
    best = (float("inf"), None, None)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        xb, yb = [], []
        for _ in range(a.batch):
            k = np.random.choice(n, p=probs)
            i = np.random.randint(0, len(tx[k]) - L)
            xb.append(tx[k][i : i + L])
            yb.append(ty[k][i + rf - 1 : i + L])
        xb = torch.stack(xb).to(device)
        yb = torch.stack(yb).to(device)
        pred = student(xb, pad_start=False)
        loss = torch.mean((pred - yb) ** 2) / (torch.mean(yb**2) + 1e-9)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == a.steps:
            e = evaluate(student, device)
            score = (e["target"] + e["holdout"]) / 2
            if score < best[0]:
                best = (score, copy.deepcopy(student.state_dict()), e)
            print(f"step {step}/{a.steps}  ESR target {e['target']:.4f}  "
                  f"holdout {e['holdout']:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    student.load_state_dict(best[1])
    student = student.cpu().eval()
    e = best[2]

    meta = {k: a2_meta[k] for k in ("name", "modeled_by", "gear_type", "gear_make",
                                    "gear_model", "tone_type", "loudness", "gain")
            if k in a2_meta}
    meta["name"] = f"{meta.get('name', 'model')} (A2->A1)"
    meta["training"] = {"distilled_from": "NAM A2",
                        "esr_target": round(e["target"], 6),
                        "esr_holdout": round(e["holdout"], 6)}
    out = {
        "version": MODEL_VERSION,
        "metadata": meta,
        "architecture": "WaveNet",
        "config": config,
        "weights": [float(v) for v in student._export_weights()],
        "sample_rate": SAMPLE_RATE,
    }
    with open(a.out, "w") as f:
        json.dump(out, f)

    worst = max(e.values())
    if worst <= GOOD_ESR:
        verdict, code = "GOOD - close match to the A2 original", 0
    elif worst <= OK_ESR:
        verdict, code = "OK - small audible differences possible", 0
    else:
        verdict, code = ("POOR - conversion failed, do not use this file.\n"
                         f"  Try again with more steps, e.g. --steps {max(20000, a.steps * 3)}"), 2
    print()
    print(f"Result: {verdict}")
    print(f"  ESR on training signal {e['target']:.4f}, on unseen test signal "
          f"{e['holdout']:.4f}  (good <= {GOOD_ESR}, ok <= {OK_ESR})")
    print(f"  -> {a.out}")
    sys.exit(code)


if __name__ == "__main__":
    main()
