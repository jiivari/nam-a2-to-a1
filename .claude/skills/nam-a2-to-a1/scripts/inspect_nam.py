#!/usr/bin/env python3
"""Show what kind of NAM capture a .nam file is (A1 or A2) and its key metadata.

Standard library only, so it runs with any Python 3.
Usage: python3 inspect_nam.py FILE.nam [MORE.nam ...]
"""
import json
import sys


def describe(path):
    try:
        d = json.load(open(path))
    except Exception as e:  # noqa: BLE001
        return f"{path}\n  ERROR: not a readable .nam file ({e})"
    arch = d.get("architecture")
    ver = str(d.get("version", "?"))
    meta = d.get("metadata") or {}
    lines = [path]
    if arch == "SlimmableContainer":
        subs = sorted(d["config"]["submodels"], key=lambda s: s["max_value"])
        sizes = ", ".join(
            f"{'Lite' if i == 0 else 'Full' if i == len(subs) - 1 else 'mid'}="
            f"{len(s['model'].get('weights', []))} weights" for i, s in enumerate(subs))
        lines.append(f"  format: A2 (version {ver}, SlimmableContainer: {sizes}) -> CONVERT")
    elif ver.startswith("0.7"):
        lines.append(f"  format: A2-style {arch} without container (version {ver}) -> CONVERT")
    else:
        detail = ""
        if arch == "WaveNet":
            ch = [l.get("channels") for l in d.get("config", {}).get("layers", [])]
            size = {(16, 8): "standard", (12, 6): "lite", (8, 4): "feather", (4, 2): "nano"}
            detail = f", {size.get(tuple(ch), 'custom')} {ch}"
        lines.append(f"  format: A1 (version {ver}, {arch}{detail}) -> no conversion needed")
    for k in ("name", "modeled_by", "gear_type", "gear_make", "tone_type"):
        if k in meta:
            lines.append(f"  {k}: {meta[k]}")
    if "loudness" in meta:
        lines.append(f"  loudness: {meta['loudness']:.1f} dB")
    tr = meta.get("training") or {}
    if isinstance(tr, dict):
        if "validation_esr" in tr:
            lines.append(f"  author's validation_esr: {tr['validation_esr']:.4f}")
        checks = (tr.get("data") or {}).get("checks") or {}
        if checks.get("passed") is False:
            lines.append("  note: the author's NAM data checks did not pass (capture may be harder to convert)")
        for k in ("esr_target", "esr_holdout"):
            if k in tr:
                lines.append(f"  converted by nam-a2-to-a1: {k} {tr[k]:.4f}")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    print("\n".join(describe(p) for p in sys.argv[1:]))
