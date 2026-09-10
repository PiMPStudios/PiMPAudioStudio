# PiMP Audio Studio Privacy Policy

> DRAFT FOR ATTORNEY REVIEW — NOT YET OPERATIVE

This draft describes how Professional Interactive Media Productions LLC
handles information in connection with the PiMP Audio Studio macOS
application. It is based on the 2026-09-09 data-flow inventory. It is not
in force until counsel reviews it, contact details are confirmed, and it is
published to users.

Do not present this document in the app or on a download page until those
steps are done.

## 1. Who We Are

Professional Interactive Media Productions LLC, a Wyoming limited liability
company, d/b/a PiMP Studios ("PiMP Studios," "we," "us," or "our") provides
PiMP Audio Studio (the "Application"), a native macOS application for local
music and sound generation, editing, dataset packaging, and optional remote
GPU training.

We are the controller of personal data you send **directly to us** (for
example a support email). We do not operate an Application backend that
stores your libraries, prompts, audio, or accounts.

For optional features that call a third-party service with credentials you
supply, that third party is an independent controller of the data it
receives. Their privacy policy and terms apply to that processing.

## 2. Scope

This policy covers the compiled PiMP Audio Studio application for macOS
(bundle ID `com.pimpstudios.PiMPAudioStudio`).

It does **not** cover:

- access to the private source repository (see the repository `LICENSE`);
- a PiMP Studios website, store, or user account, if one is added later;
- models, datasets, or cloud providers you choose, except to explain that
  the Application can send data to them at your direction.

If we add accounts, billing, analytics, or a hosted service, we will update
this policy before those features ship.

## 3. Summary

- The Application does not create a PiMP Studios account.
- After required models are on your Mac, generation, editing, local lyrics,
  stem separation, and local training run on your device.
- We do not receive your prompts, lyrics, audio, datasets, or API keys
  unless you choose to send them (for example by emailing a bug report).
- Optional features can send data to Hugging Face, GitHub, lyric AI
  providers, YouTube (via `yt-dlp` on your Mac), RunPod, Apple, or local
  tools you install. You turn those features on and provide the credentials.
- You can use the Application without those optional cloud features once
  models are downloaded.

## 4. Information We Do Not Collect In The Application

The Application does not collect or transmit to PiMP Studios:

- name, email, or login for a PiMP account (there is none);
- payment card or subscription data (no in-app store);
- advertising identifiers, analytics SDKs, or usage telemetry;
- crash reports to a PiMP server;
- your prompts, lyrics, audio, adapters, or datasets in ordinary use.

## 5. Information On Your Device

The Application stores data in the macOS sandbox, Keychain, preferences,
and an output folder you choose. Typical contents:

- downloaded model weights and adapters;
- local large-language-model weights, if you install them;
- generation history (audio files and traces that include prompt and
  lyrics);
- presets, datasets, training jobs, and editor exports;
- optional diagnostics logs;
- API keys and an optional MCP token in Keychain
  (`WhenUnlockedThisDeviceOnly`, not synced with iCloud Keychain);
- a protected sandbox copy of an SSH private key if you enable remote GPU
  automation.

This information stays on your Mac unless you export it, share it, enable
an optional network feature below, or attach it to a message you send us.

You are responsible for access to your Mac, Time Machine and other backups,
and files you place in the output folder.

## 6. Optional Third-Party Services

These paths run only if you use the related feature. PiMP Studios does not
receive the payloads. The third party's terms, retention, training, and
location rules apply. Review them before you enable the feature.

### Model and catalog downloads

The Application downloads model files and, if you refresh community
adapters, a public catalog from GitHub. Downloads may go to Hugging Face,
including gated models if you paste a Hugging Face token. We send download
requests and, when you have saved one, your token. We do not send your
audio or prompts for a model download.

### Lyric writing over an API

If you choose an API backend and save a key, the Application sends text
(a system prompt plus your prompt, style, and duration plan) to one of:

- Anthropic (Claude);
- OpenAI;
- Google (Gemini);
- xAI (Grok).

No audio file is attached to those lyric requests.

### Local command-line tools

If you enable Local CLI Access, the Application may run tools you already
installed (`claude`, `codex`, `grok`, `hf` / `huggingface-cli`). Those
tools use **your** login and may send lyric text or auth probes over their
own network path. The Application does not copy those passwords into our
Keychain. macOS sandbox exceptions may let the Application read common
install and config folders (Homebrew prefixes, Hugging Face cache, Codex,
Grok, Gemini state) so those tools can run.

### YouTube dataset import

If you import a YouTube URL and have `yt-dlp` installed, `yt-dlp` contacts
YouTube/Google from your Mac. Downloaded audio is stored in your dataset
folder. You must have the rights to use that material.

### Remote GPU training (RunPod)

If you launch remote training, the Application calls RunPod's API with your
RunPod key, may register your SSH public key on your RunPod account, and
uploads a job package over SSH to a pod you confirmed. That package can
include reviewed audio, captions, lyrics, and training settings. It does
not include API keys or your SSH private key. RunPod and the region you
pick then hold that content for as long as the pod or volume exists.
Terminate the pod after you pull artifacts if you do not want it to remain.

### Local MCP server

The MCP server is **off by default** and binds to `127.0.0.1` (this Mac
only). If you enable it, local agents on your Mac can read and write
Application data, including history prompts and lyrics. If you bind to all
interfaces, other devices on your network can reach that data; a bearer
token is then required. Do not expose MCP on an untrusted network.

## 7. Apple

PiMP Studios submits the signed Application **binary** to Apple for
notarization. That is our software, not your library.

macOS Gatekeeper may check a notarization ticket with Apple when you
launch. If you opt into Apple analytics or crash sharing in System
Settings, Apple may receive crash or diagnostic data under Apple's policy.
The Application does not embed a third-party crash or analytics SDK.

## 8. If You Contact Us

If you email us or send a bug report, we receive whatever you include
(message text, contact address, screenshots, logs, traces, or audio). Do
not send API keys, SSH private keys, or MCP tokens. See the in-app bug
report checklist.

We use that correspondence to respond, debug, and improve the Application.
Proposed retention: support mail and attachments for up to two years,
unless a longer period is required by law or a dispute. Confirm this period
with counsel before publication.

## 9. Legal Bases (EEA, UK, And Similar Laws)

Where GDPR, UK GDPR, or similar law applies, our limited first-party
processing is typically:

- **Contract / requested service** — providing the Application you
  downloaded and responding to support you request;
- **Legitimate interests** — securing the Application, diagnosing defects
  you report, and defending legal claims, balanced against your rights;
- **Legal obligation** — tax, accounting, or lawful requests;
- **Consent** — where a rule requires it for a specific optional
  disclosure.

Processing by Anthropic, OpenAI, Google, xAI, Hugging Face, GitHub,
YouTube/Google, RunPod, Apple, or a CLI vendor is under **that party's**
legal bases and policy, because you directed the Application to use their
service with your credentials.

## 10. Retention

- **On your Mac:** models, history, datasets, keys, and logs remain until
  you delete them. Optional AUDIODIAG files are pruned (about 14 days, 40
  files, 64 MB, keeping the active file) and can be cleaned from the app.
- **PiMP Studios:** we do not host your library. Support correspondence
  follows Section 8.
- **Third parties:** their retention rules apply to data you send them.

## 11. Your Choices

- Do not enable optional cloud, CLI, YouTube, RunPod, or MCP features.
- Remove API keys in Settings.
- Keep MCP off, or bound to localhost.
- Change or empty the output folder; delete history, models, and
  diagnostics.
- Revoke sandbox folder grants for provider CLIs.
- Terminate RunPod pods after pulling results.
- Use each vendor's account settings to delete data they hold.

There is no "download my library from PiMP Studios" button because we do
not host that library. You can copy files from your output folder and
Application Support container.

## 12. Your Rights

Depending on your jurisdiction, you may have rights to access, correct,
delete, restrict, port, or object to processing of personal data **we**
hold, and to withdraw consent where processing is based on consent. To
exercise those rights for data PiMP Studios holds, use the contact in
Section 20. We will respond within the time your law requires (often 30
days under GDPR/UK GDPR, 45 days under CCPA, or as otherwise required).

For data held only on your Mac, use the Application and Finder. For data
held by a third-party service you enabled, contact that service.

You may complain to a supervisory authority (for example an EU DPA, the
ICO, the Office of the Privacy Commissioner of Canada, or the OAIC).

## 13. California (CCPA/CPRA)

We do not sell personal information. We do not share personal information
for cross-context behavioral advertising. We do not use sensitive personal
information for purposes that require a CPRA opt-out.

In ordinary Application use we do not collect the CCPA categories of
personal information on our servers. If you contact us, we may receive
identifiers (email) and whatever customer records or audio files you
attach.

California residents may request to know, delete, or correct personal
information we hold, and are entitled to non-discrimination for exercising
those rights. Submit requests via Section 20. We will verify the request
by reasonable means (for example matching the email used to contact us).

## 14. Canada, Quebec, And Australia

Canadian users (including Quebec Law 25) and Australian users have access
and correction rights, and other rights under PIPEDA, Law 25, or the
Australian Privacy Principles, as applicable. The Application does not
make automated decisions that produce legal or similarly significant
effects about you. Contact us via Section 20, or your local commissioner
(OPC, Quebec CAI, or OAIC).

## 15. Children

The Application is not directed to children under 13 (or the equivalent
age of digital consent in your country, which may be 16). Do not use it
to generate, train on, or upload personal data about children. We do not
knowingly collect personal information from children. If you believe we
have received such information in a support message, contact us and we
will delete it.

## 16. International Transfers

PiMP Studios is a Wyoming limited liability company. Our mailing address is
in Sheridan, Wyoming. If you email us from another country, your message is
processed in the United States.

Optional third-party services may process data in the United States or
other countries they disclose. You initiate those transfers by enabling
the feature and using your account. Their transfer mechanisms (including
standard contractual clauses, where they offer them) are described in
their documentation.

## 17. Sensitive Content

Prompts, lyrics, and training audio can contain anything you put in them,
including voices, names, or other sensitive information. Do not submit
personal data about other people unless you have a lawful basis. Third
parties you enable may use inputs under their own policies, which can
include safety review or training; read those policies.

## 18. Automated Processing

Local "intelligence" features (intent routing, prompt enhancement, local
lyrics, captioning, transcription) run on your Mac to help you write
prompts and label audio. They do not make legal or similarly significant
decisions about you. Cloud lyric APIs, if you enable them, return text
from that vendor's models.

## 19. Changes

We may update this policy. The updated version will show a new effective
date. For material changes we will provide additional notice appropriate
to the distribution channel (for example an in-app notice or a note with
the download). Continued use after the effective date means you accept the
updated policy, except where your law requires consent.

## 20. Contact

**Controller:** Professional Interactive Media Productions LLC, d/b/a PiMP
Studios

**Privacy contact:** [support@pimp.net](mailto:support@pimp.net)

**Postal address:**

```text
30 N Gould St #65749
Sheridan, WY 82801
USA
```

**EU / UK representative:** [Appoint before targeting EU/UK users if
counsel determines GDPR Art. 27 / UK equivalent applies]

Confirm that `support@pimp.net` is a monitored mailbox before publication.

## 21. Effective Date

**Effective date:** [Publication date]

This draft is not effective until that date is filled in and the policy is
published.

## Before Publication

Counsel should, at minimum:

1. Confirm party name, d/b/a, and that `support@pimp.net` is monitored.
2. Decide whether an EU/UK representative is required.
3. Confirm support-mail retention and any state privacy addenda
   (including extra US state laws beyond California).
4. Reconcile this policy with `eula-draft.md` and any website or store
   terms.
5. Re-check the data-flow inventory if network hosts, analytics, accounts,
   or payments are added.
6. Decide how users are shown the policy (download page, first launch,
   Help).

This document is a drafting aid, not a substitute for advice from a
licensed attorney.
