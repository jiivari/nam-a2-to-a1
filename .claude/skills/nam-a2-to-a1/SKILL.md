---
name: nam-a2-to-a1
description: Converts Neural Amp Modeler (NAM) A2 captures (.nam files with model version 0.7 / "SlimmableContainer") into A1-standard .nam files using the nam-a2-to-a1 repo's distillation tools. Use this whenever someone has a .nam capture that their host can't load — e.g. a Hotone Ampero (1st gen) editor hanging on import, an older NAM plugin or pedal rejecting the file, "unsupported version", "Unrecognized architecture" — or asks to convert, downgrade or "make A1" from an A2 / A2-Lite / A2-Full capture, or wants to know whether a .nam file is A1 or A2, even if they don't mention this repo.
---

# NAM A2 → A1 conversion

This repo turns an A2 capture into an A1-standard capture by **distillation**: the A2 model (teacher) is played a training signal, and an A1 WaveNet (student) is trained to reproduce its output. It is a re-training, not a file translation, so results vary per capture and every run ends with a quality verdict.

Repo root = three levels up from this file (`.claude/skills/nam-a2-to-a1/` → repo). All commands below run from the repo root.

## 1. Check what the file is

Run the bundled inspector on each file before doing anything else. A1 files need no conversion, and the metadata helps later decisions:

```bash
python3 .claude/skills/nam-a2-to-a1/scripts/inspect_nam.py FILE.nam [MORE.nam ...]
```

- **A1** (version 0.5.x, `WaveNet`/`LSTM`): already loadable by older hosts. Tell the user no conversion is needed.
- **A2** (`SlimmableContainer`, version 0.7): convert it. One A2 file contains both the Lite and the Full submodel, and the tool uses Full. There is no separate "A2-Lite file" to look for.
- **0.7 WaveNet without a container** (e.g. a Lite part extracted on its own): can also be converted.

Before converting A2 files, and especially when there are several to choose from (e.g. L/M/R mic variants), run the **prediction** once the environments exist (step 2). It takes seconds per file and spares the user 5+ minutes of training on a capture that won't convert:

```bash
./nam-a2-to-a1 --predict FILE.nam [MORE.nam ...]
```

It compares the capture's own A2-Lite with its A2-Full. If even Lite stays close to Full, A1-standard (which is similar in size) can learn the capture. Results so far: a Lite-vs-Full ESR of ≤ 0.06 converted GOOD/OK, ≥ 0.35 converted POOR, and 0.06–0.15 is untested ground, so suggest a `--steps 1500` trial there. The thresholds rest on a few captures, so present the result as an estimate, not a guarantee. Among several variants, convert the one with the lowest score first.

## 2. Make sure the environments exist

The tool needs two Python environments (`.venv` with NAM 0.13 to read A2, `.venv-a1` with NAM 0.11 to write A1), because neither NAM version can do both. If either is missing, run `./setup.sh`. It downloads about 2 GB (PyTorch twice), so say so before running it.

## 3. Choose the training signal

The student only learns what the teacher is shown, so the signal matters:

1. **Hotone Ampero user on macOS:** use the editor's test signal at `~/Library/Application Support/com.hotone.mp380/wavs/nam_input_wav.wav`. The tool picks it up automatically. The editor creates the file at its first NAM import, and it was observed to stay unchanged afterwards. If the file is missing, the user can import any A1 `.nam` once to create it (not verified on a fresh install), or simply go on: the tool falls back to the synthetic signal, which also gave OK results. Don't claim that the editor still renders imports with this exact file; that is unconfirmed.
2. **Others:** NAM's standard `input.wav`, or any varied guitar DI recording of at least 10 s, passed with `--input`.
3. **Nothing available:** run without `--input`. A synthetic signal (pink noise and sweeps) is used. It works, but say that the result may be less accurate on real playing.

## 4. Run the conversion

```bash
./nam-a2-to-a1 --outdir OUTPUT_DIR [--input SIGNAL.wav] [--steps N] FILE.nam [MORE.nam ...]
```

- The default is a fixed 6000 training steps. That takes about 5 min per file on an Apple M-series Pro chip, about 15–20 min on a base M1/M2, and hours on CPU only. Run it in the background and tell the user roughly how long to expect.
- The output is `OUTPUT_DIR/<name> (A1).nam`. A POOR result is renamed `<name> (A1) POOR - do not use.nam`.
- `--steps 1500` is a quick feasibility check, at about a quarter of the time.

## 5. Read the verdict and act on it

Each file ends with a verdict based on ESR (error-to-signal ratio, lower is better). It is measured on the training signal and on a test signal the model never saw:

| Verdict | ESR | What to tell the user / do |
|---|---|---|
| GOOD | ≤ 0.02 | Ready to use |
| OK | ≤ 0.05 | Usable, with small audible differences possible |
| POOR | > 0.05 | Don't use it. Offer one retry with `--steps 20000` (about 3.5× longer) |

If a 20000-step run is still POOR, look at the log. When the ESR stopped falling for the last several thousand steps, the capture is too complex for A1-standard. More steps won't help. Recommend looking for an A1 version of the same amp instead (TONE3000 often has one). A useful predictor: when a capture's own A2-Lite differs a lot from its A2-Full, the capture is hard to convert. Don't promise that every capture converts. In testing, a 5150-style capture reached ESR 0.018 at 6000 steps, while an Orange Rockerverb stalled at about 0.09 even after 20000 steps.

## 6. Hotone Ampero (1st gen) specifics

These are worth mentioning when the user has an Ampero:
- The Ampero doesn't run NAM. Its editor renders the capture with its test signal and fits Hotone's own Sound Clone model, so the final sound is an approximation in any case. The editor hangs on A2 files. Converted A1 files import normally.
- Slots **6–10** belong to the AMP module. Slots **1–5** belong to FX1/FX2, and are reached by opening *Sound Clone Manage* from an FX module set to a Sound Clone. A full-rig capture works in an FX slot.
- For full-rig (`amp_cab`) captures, turn the CAB module (and the AMP module when using an FX slot) off. Watch CPU: a Sound Clone plus several other modules can cause "System Overload".
- Sound Clone slots are mono. If a capture comes as L and R files, convert one. Prefer the one with the lower `validation_esr` in its metadata (the inspector shows it).

## Etiquette

Converted files derive from someone else's capture. They're fine for the user's own use, but mention that they shouldn't be redistributed without the original author's permission.
