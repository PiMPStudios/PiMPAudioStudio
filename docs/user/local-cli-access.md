# Local CLI Access

PiMP Audio Studio can use local command-line tools for optional provider-backed
lyric and assistant workflows. The app does not store these provider passwords.
It checks whether each provider CLI is installed and signed in.

Open this screen from:

```text
Settings -> Intelligence -> Local CLI Access
```

Use the refresh button after signing in from Terminal.

## Providers

| Provider | CLI command | App auth check | Typical sign-in command |
| --- | --- | --- | --- |
| Claude | `claude` | `claude auth status` | `claude` or `claude auth login` |
| OpenAI Codex | `codex` | `codex login status` | `codex login` |
| Grok | `grok` | `grok models` | `grok login --oauth` |
| Hugging Face | `hf` or `huggingface-cli` | `hf auth whoami` | `hf auth login` |

Gemini is **not** listed under Local CLI Access. Google’s free-tier Code Assist /
`gemini` client currently reports OAuth “signed in” while lyric runs fail with
`IneligibleTierError`. Use **Settings → Intelligence → API Keys** with a
`GOOGLE_API_KEY` (or `GEMINI_API_KEY` env) for Gemini lyric writing instead.

## How The Buttons Work

- **Refresh** — reruns detection for every provider.
- **Checkmark** — runs the provider test or auth probe again.
- **Gear** — grants sandbox folder access when macOS blocks the app from reading
  a provider install folder. It does **not** sign in to the provider.

## Grok Says Test Failed

Symptom:

```text
Grok (xAI)
Test failed
Not signed in
```

Verify in Terminal:

```bash
grok --version
grok models
```

Bad output:

```text
You are not authenticated.
```

Fix:

```bash
grok login --oauth
```

If browser OAuth has trouble, use device authentication:

```bash
grok login --device-auth
```

Verify again:

```bash
grok models
```

Then return to PiMP Audio Studio and click Refresh.

Note: `grok models` may still list models while also saying
`You are not authenticated`. The important part is whether that unauthenticated
line is gone.

## Sandbox Access

PiMP Audio Studio is sandboxed. Most provider CLIs store credentials under your
home directory, such as:

```text
~/.grok
~/.gemini
~/.cache/huggingface
```

If the app shows `Sandbox access needed`, click the folder button for that row
and grant access to the suggested folder. After granting access, click Refresh.

### Write Lyrics and provider sessions

The sandbox can **read** several provider homes for sign-in checks:

```text
~/.grok   (read-only)
~/.codex  (read-only)
~/.gemini (read-write)
```

But **Write Lyrics** often needs the CLI to create temporary session/cache files.
PiMP does **not** need durable sessions for lyrics, so CLI lyric runs use an
isolated temporary HOME under the app container, seeded with only the auth
files required, then discarded. That avoids session write failures against the
real provider home.

If Write Lyrics still fails, open:

```text
~/Library/Containers/com.pimpstudios.PiMPAudioStudio/Data/Library/Application Support/PiMPAudioStudio/Logs/lyric-cli-last-failure.log
```

(or Application Support when not sandboxed) for the last CLI stdout/stderr.

### Gemini

Gemini is API-key only in PiMP (no Local CLI row). Configure a Google AI Studio
key under **API Keys**, or use Local / Claude / Codex / Grok for CLI-backed
lyrics.

## Good Debug Information To Capture

When local CLI access fails, collect:

```bash
which grok
grok --version
grok models
```

Swap `grok` for the provider that is failing. Do not paste API keys or auth
tokens into bug reports.

## Related

- [Intelligence And Lyrics](intelligence.md)
- [Troubleshooting](troubleshooting.md)
