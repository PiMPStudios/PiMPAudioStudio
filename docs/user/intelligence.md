# Intelligence And Lyrics

PiMP Audio Studio includes an intelligence layer for prompt routing, prompt
enhancement, and lyric writing. These features are optional: generation works
without them.

Open settings from:

```text
Settings -> Intelligence
```

## Local Model

A local on-device model can power:

- Intent routing for Auto mode (Music vs Sound FX vs Ambient)
- Prompt enhancement
- Local lyric writing

PiMP only supports **Llama 3–compatible MLX instruct** checkpoints (the runtime
uses a Llama 3 chat template and Llama decoder). Settings offers two stock
sizes:

| Model | Role |
| --- | --- |
| Llama 3.2 3B Instruct (4-bit) | Default — faster, less RAM |
| Llama 3.1 8B Instruct (4-bit) | Higher quality — more RAM |

There is no “add arbitrary Hugging Face repo” control for the local LLM; use
Download / Activate on the stock rows only. After enhancement or lyric writing
finishes, the app can evict the local LLM so music/SFX models regain memory
headroom before Generate.

## Prompt Enhancement

On **Generate**, use the enhance control to rewrite or strengthen the current
prompt. Enhancement returns text into the prompt field; it does not change
models, adapters, or history by itself.

The enhancer aims for a **single dense audio-caption paragraph** (genre,
instruments, voice, tempo, mood, place) suitable for ACE-Step-style generation —
not a titled production brief or mix notes.

Tips:

- Start with a clear seed idea before enhancing; keep concrete nouns you care about.
- Re-run enhancement if the result is too generic or too long.
- Cancel/stop if a run is stuck; free memory before generating music.

## Lyric Writer

The lyric writer can fill the Generate lyrics field. Choose a backend in
Intelligence settings.

Typical backends:

| Backend | Notes |
| --- | --- |
| Local (On-Device) | Uses the active local Llama model when installed |
| Claude (Anthropic) | CLI and/or API key paths |
| OpenAI / Codex-style | CLI and/or API key paths |
| Gemini (Google) | API key only (no Local CLI row) |
| Grok (xAI) | CLI and/or API key paths |

### API Keys

Optional API keys are stored in macOS Keychain via Settings. Keys are never
written into dataset packages, remote job zips, traces, or normal diagnostics.

Environment-variable fallbacks may exist for some providers when no Keychain
value is set. Prefer Keychain for the sandboxed app.

### Local CLIs

Several providers can authenticate through a local CLI instead of (or in
addition to) an API key. Manage CLI detection under:

```text
Settings -> Intelligence -> Local CLI Access
```

See [Local CLI Access](local-cli-access.md).

## What Intelligence Does Not Do

- It does not silently retrain adapters.
- It does not upload your audio to a provider unless you explicitly use a cloud
  lyric/API path that sends text prompts.
- It does not replace captions during dataset enrichment (enrichment uses its
  own annotation backends).

## MCP

When the local MCP server is enabled, tools such as `enhance_prompt` and
`write_lyrics` call the same intelligence services as the UI and return text
only—they do not mutate the open Generate fields unless the client writes them
back.

See [MCP For Users](mcp.md).

## Privacy Tips

- Review prompts/lyrics before sending them through cloud providers.
- Prefer local backends for private projects.
- Redact secrets from any shared logs or bug reports.
