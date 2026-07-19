# Bug Report Checklist

Good bug reports save time because they include the exact run context, logs, and
artifacts needed to reproduce or understand the issue.

## Do Not Include Secrets

Never share:

- RunPod API keys
- Hugging Face tokens
- OpenAI / Anthropic / Gemini / xAI keys
- SSH private keys
- Full provider account secrets
- MCP bearer tokens

If a command includes a token, redact it before sharing.

## Always Include

- PiMP Audio Studio version
- macOS version
- Mac model and memory size
- What screen/workflow you were using
- What you expected to happen
- What actually happened
- Whether the issue repeats after restarting the app

## Generation Issues

Include:

- Prompt
- Lyrics
- Seed
- Duration
- Guidance preset and scale
- Precision / steps / shift if non-default
- Active base model
- Active adapter names and gains
- Generated `.wav`
- Sibling `.trace.json`
- Newest `PiMPAudioStudio-AUDIODIAG-*.log` if verbose diagnostics were enabled

If the problem is "adapter does nothing", include both:

- Base generation with no adapter
- Adapter generation with the same seed/settings

## Training Issues

Include:

- Dataset package path or name
- Training bundle used, such as `training_bundle_vocals.json`
- Adapter kind: LoRA or LoKR
- Recipe: Narrow or Wide
- Rank/linear dim, alpha/gain settings, layer coverage, projections, epochs,
  seed, and DoRA setting
- `training_log.json`
- Newest AUDIODIAG log
- Checkpoint or adapter folder if the issue is import/inference-related

If training stalled, include the last 50–100 relevant log lines around:

```text
training/
valueAndGrad
gradClosure
training/progress
```

## Dataset Issues

Include:

- Dataset package folder
- `dataset_manifest.json`
- `captions.json`
- `package_validation.json`
- The specific source file or clip id that looks wrong
- Sampling mode, clip length, clips per file, and max clips

If clips repeat unexpectedly, include at least two repeated clip filenames and
their manifest entries.

## Remote GPU Issues

Include:

- Provider and GPU type
- Pod template or image name
- Worker version from `pimp-remote-worker doctor` when available
- The command or UI action that failed
- `remote_job.json`
- `remote_artifacts/logs.jsonl`
- `remote_artifacts/metrics.jsonl`
- `remote_artifacts/result.json`
- `remote_artifacts/command/dry_run.json` if present

If the pod was terminated, download small logs/metadata first.

## Editor Issues

Include:

- Edit mode (inpaint / repaint)
- Selection start/end if relevant
- Region prompt
- Strength and guidance
- DSP effect chain if any
- Pipeline model readiness
- Export format if export failed

## Model Download Issues

Include:

- Model display name
- Model id if visible
- Exact error text
- HTTP status code if shown
- Whether a Hugging Face token is saved
- Whether the Hugging Face license was accepted

Do not include the token itself.

## Local CLI Access Issues

Include:

- Provider row that failed
- Provider CLI version
- Exact command output from a safe status command

Examples:

```bash
claude --version
codex --version
gemini --version
grok models
hf auth whoami
```

For Grok specifically, `You are not authenticated.` means the Grok CLI is not
logged in yet. Run `grok login --oauth` or `grok login --device-auth`.

## MCP Issues

Include:

- Whether MCP is enabled
- Endpoint used
- Whether a bearer token is enabled (not the token value)
- Tool name and whether `dryRun` / `confirm` were set
- Sanitized request/response JSON

## Minimum Useful Report Template

```text
App version:
macOS:
Mac/RAM:
Workflow:
Expected:
Actual:
Repro steps:
Model:
Adapters:
Seed/settings:
Attached files:
Logs:
Notes:
```
