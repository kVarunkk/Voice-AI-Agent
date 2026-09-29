# myvoiceai

A reusable voice AI session package for real-time WebSocket voice pipelines.

## Installation

Install the package from PyPI:

```bash
pip install myvoiceai
```

Install with optional observability support:

```bash
pip install myvoiceai[observability]
```

## Quickstart

Use the `run_voice_session()` function to start a voice session. Pass API keys and session config as arguments (no required `.env`):

```python
from myvoiceai import run_voice_session

await run_voice_session(
    websocket=client_websocket,
    system_prompt="You are a helpful assistant.",
    greeting_message="Hello! How can I help you today?",
    session_id="session-001",
    model="gemini-2.5-flash",
    llm_provider_api_key="your-gemini-key",
    deepgram_api_key="your-deepgram-key",
    tracing=False,
    endpointing=1200,
    utterance_end=2500,
    stable_interim_secs=1.5,
    stable_interim_secs_no_punct=3.0,
    inactivity_timeout_seconds=30,
)
```

## Custom Tools

Pass only one tool (`echo`) with an OpenAI-style `function` schema:

```python
from myvoiceai.tools import ToolRegistry

my_registry = ToolRegistry()

async def echo(**kwargs):
    return kwargs.get("text", "")

my_registry.register(
    name="echo",
    schema={
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Echo back the text provided by the LLM.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to echo back"}
                },
                "required": ["text"],
            },
        },
    },
    impl=echo,
)

await run_voice_session(
    websocket=ws,
    tool_registry=my_registry,
    llm_provider_api_key="your-gemini-key",
    deepgram_api_key="your-deepgram-key",
)
```

## Example Server

Run the FastAPI example server (keys passed via `.env` in the example, but they can also be hardcoded or passed through config):

```bash
uvicorn myvoiceai.example.fastapi_app:app --host 0.0.0.0 --port 8000
```

Connect WebSocket clients to `ws://localhost:8000/ws/voice`.

## Key Configuration

Keys are passed directly to `run_voice_session()` or `CustomVoiceAgent`:

- `api_key` — LLM provider API key (e.g., Gemini / OpenAI)
- `deepgram_api_key` — Deepgram STT/TTS key
- `session_id` — Session identifier for tracing/logging (replaces old `interview_id`)
- `model` — Model identifier (e.g., `gemini-2.5-flash`)
- `tracing` — `True` only if `[observability]` extras are installed; default `False`

A `.env.example` is included for local convenience but is optional.

## API Reference

### `run_voice_session(websocket, ...)`
Main entry point. Parameter reference:

| Parameter | Use / Purpose |
|---|---|
| `websocket` | WebSocket connection from FastAPI / client |
| `system_prompt` | LLM system instruction (default: concise voice assistant) |
| `greeting_message` | First spoken message to user |
| `session_id` | Session identifier for tracing/logs (replaces old `interview_id`) |
| `model` | LLM model ID (e.g., `gemini/gemini-2.5-flash`) |
| `llm_provider_api_key` | LLM provider API key (required) |
| `deepgram_api_key` | Deepgram STT/TTS API key (required) |
| `stt_model` | Deepgram STT model (default `nova-2`) |
| `tts_model` | Deepgram TTS voice model (default `aura-asteria-en`) |
| `tracing` | `True` only with `[observability]` installed; sends traces to OTLP |
| `tool_registry` | `ToolRegistry` with custom `function` schemas |
| `endpointing` | Pause duration (ms) after endpointing |
| `utterance_end` | Stop recording (ms) when utterance ends |
| `stable_interim_secs` | Stable interim result delay (secs) |
| `stable_interim_secs_no_punct` | Same, when no punctuation detected |
| `inactivity_timeout_seconds` | Close session after silence (default 10) |
| `max_session_seconds` | Hard session time cap (`CustomVoiceAgent` only) |

### `CustomVoiceAgent`
Core pipeline class. Same params as `run_voice_session` plus `client_websocket`, `max_session_seconds`, `max_duration_message`, `inactivity_message`.

### `CustomVoiceAgent`
Core pipeline class. Initialize with `client_websocket` and optional `system_prompt`, `greeting_message`, `session_id`, `model`, `api_key`, `deepgram_api_key`, `tracing`, `tool_registry`.

### `ToolRegistry`
Dynamic registry for callable tools. Use `.register()` with full `function` schema, `.lookup()` to retrieve, `.schemas()` to get LLM-ready tool definitions.

## Observability / Tracing

Install extras and run the local collector stack to enable tracing:

```bash
pip install myvoiceai[observability]
```

Example infrastructure files are included in `example/`:

- `docker-compose.yml` — Jaeger (UI at `localhost:16686`) + OpenTelemetry Collector (`4318`/`4319`)
- `otel-collector-config.yaml` — Collector pipeline: OTLP → batch → Jaeger + Prometheus metrics
- `prometheus.yml` — Scrapes collector metrics at `localhost:8889`

Start the stack:

```bash
cd example
# Set OTLP endpoint in your app to http://localhost:4318 (HTTP) or localhost:4317 (gRPC)
docker-compose up -d
```

Then run the session with tracing enabled:

```python
await run_voice_session(
    websocket=ws,
    tracing=True,
    session_id="session-001",
    api_key="...",
    deepgram_api_key="...",
)
```

Jaeger UI: `http://localhost:16686` (search by `session.id`).

## WebSocket Messages

Server sends JSON over the WebSocket:

- `transcript_chunk` — incremental partial transcript (`role`: `user` or `assistant`; `turn_id`; `text`; optional `replace`)
- `transcript` — finalized transcript (`role`; `text`; `turn_id`)
- `turn` — completed turn (`turn_id`; `timestamp`; `user`; `assistant`; `interrupted`: bool)

Client should listen for these to update UI/state.

## Notes

- The package does **not** include infrastructure files like `docker-compose.yml` or `prometheus.yml`. These are maintained separately.
- Observability (OpenTelemetry) is optional via `pip install myvoiceai[observability]`. Without it, `tracing=False` runs normally.
- `.env` is excluded from the package for security. Only `.env.example` ships for reference.
- All tool schemas must use the OpenAI-style `{"type": "function", "function": {...}}` format.
