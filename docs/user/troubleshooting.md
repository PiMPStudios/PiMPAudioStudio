# Troubleshooting

Use this page when something does not work and you need a quick path from
symptom to verification to fix.

## Local CLI Access

### Grok shows "Test failed" or "Not signed in"

Verify:

```bash
grok models
```

If the output includes `You are not authenticated.`, fix with:

```bash
grok login --oauth
```

or:

```bash
grok login --device-auth
```

Then click Refresh in **Settings → Intelligence → Local CLI Access**.

### Provider shows "Sandbox access needed"

Click the folder-badge icon for that provider row and grant access to the
suggested folder. Then click Refresh. That control grants folder access only; it
does not sign in to the provider.

### Grok Write Lyrics exits with status 1

Common causes:

1. Not signed in (`grok models` shows “You are not authenticated”).
2. Older app builds tried to write Grok sessions under real `~/.grok`, which the
   sandbox may block even when sign-in checks pass.

Fix:

```bash
grok login --oauth
# or: grok login --device-auth
grok models
```

Then use a build that runs Grok lyric writes with an isolated temporary HOME
(so sessions do not need write access to real `~/.grok`). If the red error
includes a Grok message, that text is the authoritative failure reason.

### Gemini is missing from Local CLI Access

Expected. Gemini lyric writing is API-key based in PiMP. Add a key under
**Settings → Intelligence → API Keys**, or pick another lyric backend.

## Model Downloads

### Hugging Face returns 401 or 403

Common causes:

- No Hugging Face token saved
- Token expired or lacks access
- Gated model license not accepted on Hugging Face

Fix:

1. Open **Settings → Models**
2. Save a Hugging Face token
3. Visit the model page and accept the license
4. Retry the download

### Model says Ready but generation fails

1. Confirm the DiT, **VAE + Text Encoder**, and any required LM / Stable Audio
   models are Ready.
2. If a download was interrupted, delete and redownload the affected model.
3. Release GPU memory, restart the app, enable verbose diagnostics, and retry.
4. Check for Soft Late + XL warnings if quality is the issue rather than a hard
   failure.

## Generation Quality

### Output is weak, noisy, or ignores the prompt

- Try Constant guidance on XL models.
- Avoid Soft Late on XL unless intentionally A/Bing.
- Use a clearer prompt; try Enhance once.
- Confirm pipeline is ACE-Step for music and Stable Audio for SFX.
- Fix seed and compare a known-good preset.

### Cover stays too close / too far from source

Adjust cover strength. Lower stays closer to the recording; higher gives more
creative freedom. Keep seed fixed while comparing.

## Adapters

### Adapter imports but sounds like it does nothing

Check:

- Compatible with the selected base model variant?
- Gain at least `1.0x`?
- Verbose logs show `adapter/effect`?
- Same seed/prompt/lyrics/duration/settings for A/B?

Try base → `1.0x` → `1.6x` → `2.0x`. If only high gain is audible, training
alpha/rank/recipe may be weak.

### Stack sounds broken

Remove the stack and test adapters one at a time. Do not mix LoRA and LoKR while
debugging.

## Local Training

### Training seems hung at the first step

Check the newest AUDIODIAG log for:

```text
training/trainer
valueAndGrad
gradClosure
firstValueAndGrad
training/progress
```

Watch memory pressure in Activity Monitor. If the app is responsive, pause/stop
to save a checkpoint. If not, collect the log before force quitting.

### Loss is flat

Flat loss is not automatically a failure for adapter training, especially with
flow-matching objectives and small parameter counts. Judge checkpoints by audio:

1. Same prompt/lyrics/seed/duration/base model
2. Base vs checkpoint
3. A few gains
4. Prefer the best-sounding checkpoint, not only lowest train loss

### Out of memory during training

- Use Narrow recipe
- Lower rank / layer count / projection coverage
- Release generation models first
- Prefer remote GPU for Wide XL recipes

## Dataset Enrichment

### Automatic falls back to lightweight

Native ACE packs missing, invalid, or Mac below memory policy. Install packs
under **Settings → Models → Dataset Annotation (Advanced)** or force Lightweight
Native.

### Captions look like filenames

Filename placeholders are intentional before enrichment/review. Run enrichment
or edit captions manually in Dataset Review.

## Remote GPU Training

### Pod launched but worker command is missing

Stock RunPod images do not include `pimp-remote-worker`. Use **Upload** (or the
app's upload command) to copy the job zip and portable worker, then Start /
install.

### First Start looks hung

Initial environment setup can take a long time (often tens of minutes). Use
**Monitor** for stage progress rather than terminating early.

### Remote run fails before training starts

Dry run:

```bash
pimp-remote-worker train /workspace/jobs/my-job.zip --dry-run
```

Inspect:

```text
remote_artifacts/command/dry_run.json
remote_artifacts/logs.jsonl
remote_artifacts/result.json
```

### Training finished but no adapter in the app

Download `remote_artifacts/final_adapter/` or a checkpoint folder and import via
adapter settings, or use **Pull & Import**.

### Pod is still billing

**Terminate** the pod after downloading artifacts. Do not leave unused pods
running or paused with storage attached.

### Upload disabled

Configure the SSH private key in the Remote GPU panel and ensure a job zip is
selected/remembered.

## Editor

### Inpaint button disabled

Need loaded audio, non-empty region prompt, and (for inpaint) a selection.

### DSP did nothing

Parameterized effects require Apply. Selection-only effects require a selection.

### Export failed

Try WAV to a user-selected folder. Check free disk and permissions.

## MCP

### Client cannot connect

Confirm MCP is enabled in Settings, endpoint is `127.0.0.1`, and any token
matches. Restart the app after toggling the server.

### Tool returns dry-run only

Many tools default to `dryRun: true`. Pass `dryRun: false` plus `confirm: true`
(or approve elicitation) for real mutations / billable launches.

## Still Stuck

Use [Bug Report Checklist](bug-report-checklist.md) and attach redacted logs.
