# Diagnostics And Logs

PiMP Audio Studio can write detailed audio and training breadcrumbs when verbose
diagnostics are enabled. These logs are useful for comparing generation runs,
checking whether adapters are active, and debugging training or remote worker
failures.

## Enable Verbose Diagnostics

Open:

```text
Generate -> Advanced -> Verbose audio diagnostics
```

When enabled, the app writes `[AUDIODIAG]` lines to the Xcode console and to a
timestamped log file in the app diagnostics folder.

## Log Location

The sandboxed app writes diagnostics under its Application Support directory:

```text
~/Library/Containers/com.pimpstudios.PiMPAudioStudio/Data/Library/Application Support/PiMP Audio Studio/Diagnostics/
```

Typical file name:

```text
PiMPAudioStudio-AUDIODIAG-2026-07-01_12-00-00-000-ABCDEF12.log
```

The Generate view also shows the active log path beside the diagnostics toggle.

## What To Capture

For generation issues:

- The newest `PiMPAudioStudio-AUDIODIAG-*.log`.
- The generated `.wav`.
- The sibling `.trace.json`, if present.
- The prompt, lyrics, seed, duration, model, and adapter stack.

For training issues:

- The newest `PiMPAudioStudio-AUDIODIAG-*.log`.
- The training run folder.
- `training_log.json`, if present.
- Dataset package path.
- Adapter recipe and settings.

For remote GPU issues:

- `remote_artifacts/logs.jsonl`.
- `remote_artifacts/metrics.jsonl`.
- `remote_artifacts/result.json`.
- `remote_artifacts/command/dry_run.json`.
- The generated `remote_job.json`.

## Useful Things To Search

Generation:

```text
generation/
adapter/effect
cond/
diffusion/
vae/
```

Training:

```text
training/
training/progress
training/preprocess
training/checkpoint
gradNorm
updateNorm
valLoss
```

Remote GPU:

```text
runpod/
setup
dataset
preprocess
train
complete
error
```

## Privacy

Logs can include prompts, lyrics, local paths, model names, adapter names, and
dataset names. Review logs before sharing them publicly.
