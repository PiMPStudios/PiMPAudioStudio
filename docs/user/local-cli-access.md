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
| Gemini | `gemini` | local OAuth settings in `~/.gemini` | `gemini` |
| Grok | `grok` | `grok models` | `grok login --oauth` |
| Hugging Face | `hf` or `huggingface-cli` | `hf auth whoami` | `hf auth login` |

## How The Buttons Work

- Refresh: reruns detection for every provider.
- Checkmark: runs the provider test or auth probe again.
- Gear: grants sandbox folder access when macOS blocks the app from reading a
  provider install folder. It does not sign in to the provider.

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

Note: `grok models` may still list models while also saying `You are not
authenticated`. The important part is whether that unauthenticated line is gone.

## Sandbox Access

PiMP Audio Studio is sandboxed. Most provider CLIs store credentials under your
home directory, such as:

```text
~/.grok
~/.gemini
~/.cache/huggingface
```

If the app shows `Sandbox access needed`, click the gear button for that row and
grant access to the suggested folder. After granting access, click Refresh.

## Good Debug Information To Capture

When local CLI access fails, collect:

```bash
which grok
grok --version
grok models
```

Swap `grok` for the provider that is failing. Do not paste API keys or auth
tokens into bug reports.
