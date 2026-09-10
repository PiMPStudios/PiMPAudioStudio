# PiMP Audio Studio Privacy Data-Flow Inventory

**Status:** working inventory for a future privacy policy. Not a privacy
policy. Not legal advice. Counsel should review before public or paid
distribution.

**As of:** 2026-09-09 (code and docs in this repository).

**Product:** PiMP Audio Studio (`com.pimpstudios.PiMPAudioStudio`), a native
sandboxed macOS 15+ app published by Professional Interactive Media Productions
LLC.

## How To Read This

PiMP Audio Studio does **not** create a PiMP user account, does **not** talk to
a PiMP-operated backend, and does **not** include first-party analytics, ads,
crash SDKs, or CloudKit.

Most work stays on the Mac after models are downloaded. Network use is for
model/catalog downloads and **optional** user-directed services (Hugging Face,
lyric APIs, YouTube via `yt-dlp`, RunPod, local MCP clients).

When a third party receives data, it is because the user turned that feature on
and supplied their own token, CLI login, URL, or cloud account. Those companies
are not PiMP subprocessors in the usual SaaS sense; they are independent
services the app can call at the user's direction. A published privacy policy
still needs to name them.

## 1. What PiMP Studios Does Not Collect In-App

The app does not collect or transmit to PiMP Studios:

- email, name, or account credentials
- payment or subscription data (no in-app store, no Stripe)
- usage analytics, funnels, or advertising IDs
- crash reports to a PiMP server
- prompts, lyrics, audio, datasets, or adapters

Support email, a future website, notarization of the **binary**, and Apple's
own crash/Gatekeeper channels are outside this in-app inventory.

## 2. What Stays On The Mac

These pipelines run locally after required weights are on disk. Content is not
sent to PiMP or to a cloud API unless a separate optional path below is used.

- Music generation (ACE-Step / MLX)
- Sound-effects generation (Stable Audio Open, when installed)
- Cover-song generation (source audio encoded and diffused locally)
- Prompt enhancement and Auto-mode intent routing (local Llama, when
  installed)
- Local lyric writing (same local Llama)
- Editor inpaint/repaint, DSP, export
- Stem separation (Demucs MLX)
- Dataset packaging, review, local enrichment (captions, transcription,
  alignment)
- Local LoRA / LoKR training
- History, presets, traces, diagnostics files

## 3. Local Storage Map

Sandbox container Application Support (typical):

```text
~/Library/Containers/com.pimpstudios.PiMPAudioStudio/Data/
Library/Application Support/PiMP Audio Studio/
```

- `models/` — downloaded weights and adapters. Leaves the Mac only if the
  user later uploads an adapter or packs files into a remote job.
- `LLM/` — local Llama instruct weights. Stays on the Mac.
- `presets/` — saved generation presets. Stays on the Mac.
- `Diagnostics/` — optional AUDIODIAG logs. Leave the Mac only if the user
  attaches them to a bug report.
- `SourceAudioImports/` — imported source audio copies. Stay unless the user
  exports or packs them.
- `TrainingDatasets/` — dataset packages (also under the user output
  folder). Leave the Mac if packed into a RunPod job zip.
- User-chosen output directory — history WAVs, `.trace.json`, training
  jobs. Same outbound rule as datasets.
- Keychain service `com.pimpstudios.pimpaudiostudio.apikeys` — API keys and
  the MCP bearer token (`mcp-bearer-token`). Keys go out only as
  `Authorization` to the matching provider. Accessibility is
  `WhenUnlockedThisDeviceOnly` (not iCloud-synced).
- UserDefaults — output-folder bookmark, UI sizes, adapter selection,
  RunPod launch prefs, MCP bind/port/enabled, verbose-diagnostics flag. No
  secrets (legacy MCP token in defaults is migrated into Keychain).
- Sandbox `~/.ssh/id_ed25519` (managed copy) — RunPod SSH private key,
  mode `0700`/`0600`. Public key may be registered with RunPod. Private key
  is used for `ssh`/`scp` to the user's pod and is never placed in the job
  zip.
- Security-scoped bookmarks — output directory and optional provider CLI
  folders. Local access grants only.

History `.trace.json` files store **prompt, original prompt, and lyrics** plus
model/adapter snapshots, seeds, and timing. They stay in the output folder
unless the user shares them.

Verbose AUDIODIAG logs are tensor/timing breadcrumbs, not a full prompt
archive. Bug-report docs still tell users to redact secrets and to expect
prompts/lyrics in generation reports they choose to send.

## 4. Secrets

- Hugging Face token — Keychain (`huggingface`) or
  `HUGGINGFACE_API_KEY`. Used for gated Hub downloads. Must not appear in
  model folders, zips, traces, or logs.
- Anthropic / OpenAI / Google / xAI API keys — Keychain or env. Used for
  optional lyric HTTP APIs. Must not appear in packages, MCP resources, or
  logs.
- RunPod API key — Keychain (`runpod`) or `RUNPOD_API_KEY`. Used for
  `https://api.runpod.io/v2`. Must not appear in job zips.
- RunPod SSH private key — protected sandbox copy. Used for `ssh`/`scp` to
  the pod. Must not appear in job zips or logs.
- MCP bearer token — Keychain. Used for optional/LAN MCP auth. Must not
  appear in bug reports.

CLI logins (`claude`, `codex`, `grok`, `hf`) live in those tools' own state
(for example `~/.codex/`, `~/.grok/`, Hugging Face cache). The app probes them
through sandbox exceptions or a user-granted folder; it does not copy those
passwords into Keychain.

## 5. Outbound Network (User Or Setup Directed)

Default entitlements allow client **and** server networking. Server is for
optional MCP. There is no PiMP API host.

### 5.1 Hugging Face (`huggingface.co`)

**When:** first-launch essential models; Settings → Models; adapter import;
optional annotation packs; Demucs weights; local LLM weights; gated Stable
Audio.

**Auth:** optional Bearer token from Keychain.

**Sent:** download requests, repo/revision/file path, token if present. Not
prompts, not user audio.

**Received:** model weights and small JSON configs.

Known Hub sources in code (not a license audit):

- `ACE-Step/*` music, VAE, LM
- `stabilityai/stable-audio-open-1.0` (gated)
- `mlx-community/Llama-3.2-3B-Instruct-4bit` and
  `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`
- `DaRealDahoodie/pimp-audio-ace-captioner-4bit-v1` and
  `...-transcriber-4bit-v1`
- `aufklarer/Qwen3-ASR-*` and `aufklarer/Qwen3-ForcedAligner-*` via
  speech-swift
- User-pasted Hugging Face adapter URLs (allowlisted hosts
  `huggingface.co` / `hf.co`)

The sandbox may also **read** `~/.cache/huggingface/` if the user already has
a Hub cache.

### 5.2 GitHub (community adapter catalog)

**When:** user refreshes community adapters.

**URL:**
`https://raw.githubusercontent.com/PiMPStudios/pimp-adapters/main/manifest.json`

**Sent:** HTTP GET. A 404 is treated as an empty catalog.

**Note:** review-followups record that PiMP will not run a signed adapter
catalog. Users import a Hub URL they chose. Treat catalog fetch as optional
metadata, not as PiMP hosting user content.

### 5.3 Lyric APIs (optional)

Triggered only from Intelligence / Write Lyrics when that backend is selected
and a key exists. Payload is a **system prompt plus user prompt/style/duration
plan** (text). No audio files.

- Anthropic Claude — `https://api.anthropic.com/v1/messages`, header
  `x-api-key`.
- OpenAI — `https://api.openai.com/v1/chat/completions`, Bearer token.
- xAI Grok — `https://api.x.ai/v1/chat/completions`, Bearer token.
- Google Gemini — `generativelanguage.googleapis.com` stream endpoint for
  `gemini-2.0-flash`, header `x-goog-api-key`.

Each provider's retention, training, and privacy terms apply. PiMP does not
see the request.

### 5.4 Local provider CLIs (optional)

Settings → Intelligence → Local CLI Access can run `claude`, `codex`, `grok`,
or `hf` / `huggingface-cli` for auth probes and lyric writing. Gemini CLI is
**not** registered; Gemini lyrics are API-key only.

Those processes use the user's existing login and may send the lyric prompt to
the same vendors over the CLI's own network path. The app does not store CLI
passwords.

Sandbox exceptions exist for `/opt/homebrew/`, `/usr/local/`, `/opt/local/`,
`~/.cache/huggingface/`, `~/.codex/`, `~/.grok/` (read), and `~/.gemini/`
(read/write). Audit these before any App Store attempt; they are why direct
Developer ID distribution is the current path.

### 5.5 YouTube dataset import (optional)

**When:** Datasets → YouTube import, if `yt-dlp` is installed.

**Sent:** playlist/channel/watch URL to YouTube/Google via `yt-dlp`
(`--no-cache-dir`). Video IDs are sanitized; only https YouTube hosts are
allowed.

**Stored:** downloaded audio, chunked clips, and caption text in the dataset
output folder.

**Not sent to PiMP.** The user is responsible for rights to the source.

### 5.6 RunPod remote GPU (optional, billable)

**When:** Training → Remote GPU, after explicit confirmation.

**API:** `https://api.runpod.io/v2` (pods, templates, SSH key registration,
terminate).

**SSH/SCP:** job zip + portable `remote-worker/` to the user's pod; later pull
of `remote_artifacts`.

**Job zip contains:** reviewed dataset package (audio clips, captions, lyrics,
manifests) plus `remote_job.json` and worker instructions. **Does not
contain** API keys or the SSH private key.

**Also sent:** derived SSH **public** key merged into the RunPod account key
set (does not wipe existing keys).

RunPod (and the GPU region the user picked) then holds user audio and labels
for the life of the pod and any volume the user leaves running. Terminate is
destroy, not pause-with-storage, and the UI warns about data loss.

### 5.7 MCP (optional, off by default)

Default bind: `127.0.0.1:38765`. Loopback only until the user selects all
interfaces.

Tools can read/write local generation, history (including prompts/lyrics),
datasets, training, and RunPod **on this Mac**. They do not proxy content to
PiMP.

If bind is `0.0.0.0`, a bearer token is **required**. LAN clients can then
reach local libraries. Prefer trusted networks only.

API keys and SSH private keys are not writable through MCP.

## 6. Apple Channels (Not App Telemetry)

- Developer ID notarization — PiMP submits the **signed app binary**, not
  user libraries. Apple's terms apply.
- Gatekeeper / stapled ticket — macOS may check notarization with Apple at
  launch. Apple's terms apply.
- macOS crash reports — only if the user opted into Share with App
  Developers / Analytics. Apple (and, if shared, the crashing app's
  developer).
- Unified logging (`os.Logger`) — on-device. May appear in a sysdiagnose
  the user exports.

No Crashlytics, Sentry, or similar SDK is in the project.

## 7. Data Categories (For A Future Policy)

Useful GDPR/CCPA labels for counsel, mapped to **this app**, not a hosted
SaaS:

- Identifiers — none in-app. Keychain holds third-party tokens the user
  pasted. PiMP does not receive them.
- User content — prompts, lyrics, audio, datasets, adapters, listening
  notes. PiMP does not receive them unless the user emails a bug report.
- Device / app prefs — output path bookmark, UI state, MCP settings. PiMP
  does not receive them.
- Diagnostics — AUDIODIAG files and traces. PiMP does not receive them
  unless attached by the user.
- Commercial cloud — RunPod job packages and SSH public key. Goes to
  RunPod, not PiMP.
- AI prompts — lyric/enhance text. Local or the user's AI vendor, not
  PiMP.
- Sensitive data — possible inside prompts, lyrics, or training audio
  (voices, children, health, and similar). The app cannot know. A policy
  should warn users not to upload others' personal data without a legal
  basis.

No sale of personal information. No advertising. No profiling for ads.

## 8. Retention (On The User's Mac)

PiMP has no cloud retention of user libraries. On-device:

- Models and user output stay until the user deletes them
- AUDIODIAG: last 14 days, at most 40 files, about 64 MB, plus the active
  file; **Clean logs** runs the same prune
- Corrupt indexes are renamed `.corrupt` rather than silently deleted
- Third-party vendors retain API/CLI/cloud data under **their** terms
- RunPod retention follows the pod/volume the user leaves running

## 9. User Controls

- Never sign in to PiMP (there is no account)
- Skip all optional network features; generation still works after local
  model download
- Remove API keys in Settings
- Disable MCP (default off)
- Bind MCP to localhost; keep token off unless LAN bind forces it
- Choose not to use YouTube import, community catalog, or RunPod
- Change or clear the output directory
- Delete history, datasets, models, diagnostics
- Revoke sandbox folder grants for provider CLIs
- Terminate RunPod pods after pulling artifacts

There is no in-app "download my data from PiMP" because PiMP does not host
that data. A policy should still explain how to delete local files and how to
exercise rights with Anthropic, OpenAI, Google, xAI, Hugging Face, GitHub,
YouTube/Google, RunPod, and Apple.

## 10. Gaps Before A Publishable Privacy Policy

1. **Counsel review** of this inventory, `eula-draft.md`, and
   `privacy-policy-draft.md` (not operative until published).
2. **Contact identity:** `support@pimp.net` and mailing address (30 N Gould
   St #65749, Sheridan, WY 82801, USA) are filled in the drafts. Formation
   is a Wyoming LLC. Confirm whether an EU Art. 27 representative is needed
   if EU users are targeted.
3. **Website / future account:** if pimp.net or billing appears, this
   inventory is incomplete (that would add first-party account, payment, and
   hosting subprocessors).
4. **Bug reports:** if users email traces/WAVs, that becomes first-party
   processing with a retention rule.
5. **Sandbox exceptions** and MCP `0.0.0.0` should be described in plain
   language (LAN exposure; CLI home-folder reads).
6. **Model/dataset licenses** (Hugging Face gates, YouTube ToS, generated
   output) are adjacent to privacy but belong in EULA / model notices too.
7. **Children:** the app is not designed for under-13; confirm COPPA/GDPR-K
   stance with counsel.
8. **Keep this file in sync** when adding a network host, analytics, account,
   or paid store.

## 11. Suggested Policy Framing (For Counsel, Not Copy-Paste)

Professional Interactive Media Productions LLC provides PiMP Audio Studio as
on-device software. We do not operate an application backend that stores your
libraries. Optional features send data to third parties you choose, using
credentials you supply. Those third parties are independent controllers of
the data they receive. If you contact us for support and attach files, we
become a controller of that correspondence.

That framing matches the code as of this inventory. It is not a substitute
for a reviewed privacy policy.

## Related Documents

- [Legal README](README.md)
- [EULA draft](eula-draft.md) (not operative)
- [Privacy policy draft](privacy-policy-draft.md) (not operative)
- [Storage, sandbox, security](../developer/storage-sandbox-security.md)
- [Release packaging](../developer/release-packaging.md)
- [User diagnostics](../user/diagnostics-and-logs.md)
- [User MCP](../user/mcp.md)
- [Remote GPU training](../user/remote-gpu-training.md)
- [Bug report checklist](../user/bug-report-checklist.md)
