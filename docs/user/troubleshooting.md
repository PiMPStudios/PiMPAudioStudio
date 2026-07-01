# Troubleshooting

Use this page when something does not work and you need a quick path from
symptom to verification to fix.

## Local CLI Access

### Grok shows "Test failed" or "Not signed in"

Verify:

```bash
grok models
```

If the output includes this line, the CLI itself is not authenticated:

```text
You are not authenticated.
```

Fix:

```bash
grok login --oauth
```

If OAuth does not complete:

```bash
grok login --device-auth
```

Then click Refresh in:

```text
Settings -> Intelligence -> Local CLI Access
```

### Provider shows "Sandbox access needed"

Click the gear icon for that provider row and grant access to the suggested
folder. Then click Refresh.

The gear button grants folder access only. It does not sign in to the provider.

## Model Downloads

### Hugging Face returns 401 or 403

Common causes:

- No Hugging Face token is saved.
- The token is expired or lacks access.
- The model is gated and the license has not been accepted on Hugging Face.

Fix:

1. Open `Settings -> Models`.
2. Save a Hugging Face token.
3. Visit the model page on Hugging Face and accept the required license.
4. Retry the download.

### Model says ready but generation fails

Try:

1. Open `Settings -> Models`.
2. Confirm the required ACE-Step model, shared components, and LM are present.
3. If a download was interrupted, delete and redownload the affected model.
4. Enable verbose diagnostics and retry.

## Adapters

### Adapter imports but sounds like it does nothing

Check:

- Is the adapter compatible with the selected base model variant?
- Is adapter gain set to `1.0x` or higher?
- Does verbose diagnostics show `adapter/effect` lines?
- Are you comparing with the same seed, prompt, lyrics, duration, and settings?

Try:

1. Generate with no adapter.
2. Generate with the adapter at `1.0x`.
3. Generate with the adapter at `1.6x` or `2.0x`.
4. Use Trace Compare or training audition snapshots to A/B the results.

If the adapter only becomes audible at high gain, the training alpha/rank or
remote recipe may have produced a weak adapter.

## Local Training

### Training seems hung at the first step

Check:

- The newest `PiMPAudioStudio-AUDIODIAG-*.log`.
- Whether the log reaches `training/progress`.
- Memory pressure in Activity Monitor.

Useful log terms:

```text
training/trainer
valueAndGrad
gradClosure
firstValueAndGrad
```

If the app is still responsive, pause/stop training to save a checkpoint. If it
is not responsive, collect the log before force quitting.

### Loss is flat

Flat loss is not automatically a failure. Adapter training loss can plateau
quickly, especially with flow-matching objectives and small adapter parameter
counts.

Judge checkpoints by audio:

1. Use the same prompt, lyrics, seed, duration, and base model.
2. Compare base vs checkpoint.
3. Try a few gains.
4. Prefer the checkpoint that sounds best, not simply the lowest train loss.

## Remote GPU Training

### Pod launched but the worker command is missing

A stock RunPod PyTorch image does not include `pimp-remote-worker`.

Use the app's upload command first. It copies both the job zip and the portable
worker folder. Then use the extract/run command.

Manual install:

```bash
bash /workspace/pimp-remote-worker-src/install.sh
source /workspace/pimp-remote-worker/env.sh
pimp-remote-worker doctor
```

### Remote run fails before training starts

Run a dry run:

```bash
pimp-remote-worker train /workspace/jobs/my-job.zip --dry-run
```

Check:

```text
remote_artifacts/command/dry_run.json
remote_artifacts/logs.jsonl
remote_artifacts/result.json
```

### Training finished but no adapter is visible in the app

Download one of these folders from the pod:

```text
remote_artifacts/final_adapter/
remote_artifacts/checkpoints/<checkpoint-folder>/
```

Then import it through adapter settings as a local PEFT/Side-Step folder.

### Pod is still billing

Terminate the pod from PiMP Audio Studio or the RunPod console after downloading
artifacts. Do not leave unused pods running or paused with storage attached.
