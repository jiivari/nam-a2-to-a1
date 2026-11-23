# nam-a2-to-a1

Convert [Neural Amp Modeler](https://github.com/sdatkinson/neural-amp-modeler) **A2** captures into **A1-standard** `.nam` files, so hardware and software that only understand A1 can use them.

It was built for the 1st-generation **Hotone Ampero**. Its editor (V1.5.0) hangs when importing an A2 file, but A1 files converted with this tool import and play fine.

## How it works

NAM A2 files use a new container format (`SlimmableContainer`, model version 0.7) and new layer types. Older loaders can't read them. This tool doesn't translate the file. It *re-trains* an A1 model to sound like the A2 model, a process called distillation:

1. **Render the teacher.** A training signal is played through the A2 model's full-size submodel at three input levels. This uses `neural-amp-modeler` 0.13, which reads A2.
2. **Train the student.** An A1-standard WaveNet learns to reproduce the A2 output. This uses `neural-amp-modeler` 0.11, which writes A1 files in model version 0.5.4. The best checkpoint is picked on a held-out 10 s section of the signal.

The tool needs two NAM versions because 0.13 can't load A1 configs and 0.11 can't load A2.

On an Apple M-series Mac one conversion takes about 5 minutes (6000 steps). On our test capture (a 5150-style high-gain amp+cab), the validation ESR against the A2 original was about 0.018 from random initialization and 0.014 when warm-started from an A1 capture of a similar amp (`--template`).

## Setup

Requirements: Python 3.10+ and about 2 GB of disk space for two PyTorch environments.

```bash
git clone https://github.com/jiivari/nam-a2-to-a1.git
cd nam-a2-to-a1
./setup.sh
```

## Usage

```bash
./nam-a2-to-a1 --outdir converted "My Amp A2.nam"
```

| Option | Default | Description |
|---|---|---|
| `--input WAV` | the Ampero Editor's test signal (macOS) | Training signal (any sample rate, at least 30 s) |
| `--outdir DIR` | `.` | Where the converted `… (A1).nam` files go |
| `--steps N` | 6000 | Training steps |
| `--template A1.nam` | random init | Warm-start from an existing A1-standard capture |

### Training signal

**Ampero users on macOS** don't need to pass anything. When the Ampero Editor imports a NAM file, it writes its own test signal to
`~/Library/Application Support/com.hotone.mp380/wavs/nam_input_wav.wav`. The tool uses that file by default. The editor then renders the converted capture with the same signal, so the model is trained on exactly what the editor will feed it.

Everyone else: pass `--input` with a varied guitar or DI signal of at least 30 s. NAM's standard training input works well. This repository doesn't ship any audio.

## Notes

- Some accuracy is lost, because A1-standard is a smaller network than A2-full. If the capture's author also publishes an A1 version, use that instead.
- The Ampero doesn't run NAM itself. Its editor converts every NAM file into Hotone's own Sound Clone model, so the final accuracy depends on that conversion. On the 1st-gen Ampero, only Sound Clone slots 6–10 accept user files.
- Respect capture authors: convert captures for your own use, and don't redistribute converted files without the author's permission.

## License

MIT. Uses [neural-amp-modeler](https://github.com/sdatkinson/neural-amp-modeler) (MIT). Not affiliated with Hotone or TONE3000.
