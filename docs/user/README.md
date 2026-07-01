# PiMP Audio Studio User Docs

These docs explain how to use and troubleshoot PiMP Audio Studio from the point
of view of someone running the app, not someone changing the code.

The current source of truth is this `docs/user/` folder. For public
distribution, these pages should be bundled into the app or rendered from an
in-app Help window so users do not need access to the source repository.

## Start Here

- [First Launch And Model Setup](model-setup.md) - required model components,
  Hugging Face tokens, storage paths, and first-launch checks.
- [Datasets And Training](datasets-and-training.md) - building reviewed
  datasets, choosing training bundles, and starting local or remote training.
- [Adapters](adapters.md) - import formats, base-model compatibility, gain,
  stacking, and auditioning.
- [Local CLI Access](local-cli-access.md) - Claude, Codex, Gemini, Grok, and
  Hugging Face sign-in checks.
- [Remote GPU Training](remote-gpu-training.md) - RunPod setup, package upload,
  worker install, training, artifacts, and cleanup.
- [Diagnostics And Logs](diagnostics-and-logs.md) - where logs live, what to
  capture, and how to read the high-value breadcrumbs.
- [Troubleshooting](troubleshooting.md) - symptom-first fixes for common issues.
- [Bug Report Checklist](bug-report-checklist.md) - what to collect for
  generation, training, dataset, remote GPU, model, and CLI issues.

## What Belongs Here

User docs should answer:

- What does this feature do?
- What should I click?
- What command can I run to verify it?
- What does good output look like?
- What should I collect when something fails?

Internal design notes, research, and long-term roadmap files should stay in the
top-level `docs/` folder or `docs/developer/`.
