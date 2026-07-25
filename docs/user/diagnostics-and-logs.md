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

When enabled, the app writes `[AUDIODIAG]` lines to the Xcode console (when run
from Xcode) and to a timestamped log file in the app diagnostics folder.

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

### Log retention

The app prunes AUDIODIAG logs automatically on launch (and when a new log file
opens). Defaults:

- keep logs from the last **14 days**
- keep at most **40** files
- keep at most about **64 MB** total
- never delete the **currently active** log

Use **Generate → Advanced → Clean logs** to run the same cleanup immediately.

## What To Capture

### Generation issues

- Newest `PiMPAudioStudio-AUDIODIAG-*.log`
- Generated `.wav`
- Sibling `.trace.json` if present
- Prompt, lyrics, seed, duration, model, adapter stack and gains

### Training issues

- Newest AUDIODIAG log
- Training run folder
- `training_log.json` if present
- Dataset package path
- Adapter recipe and settings

### Remote GPU issues

- `remote_artifacts/logs.jsonl`
- `remote_artifacts/metrics.jsonl`
- `remote_artifacts/result.json`
- `remote_artifacts/command/dry_run.json`
- Generated `remote_job.json`

### Dataset issues

- Package folder
- `dataset_manifest.json`
- `captions.json`
- `package_validation.json`

## Useful Search Terms

### Generation

```text
generation/
adapter/effect
cond/
diffusion/
vae/
```

### Training

```text
training/
training/progress
training/preprocess
training/checkpoint
gradNorm
updateNorm
valLoss
valueAndGrad
```

### Remote GPU / RunPod

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
dataset names. Review logs before sharing them publicly. Never paste API keys,
SSH private keys, or Hugging Face tokens into bug reports.

## Related

- [Bug Report Checklist](bug-report-checklist.md)
- [Troubleshooting](troubleshooting.md)
