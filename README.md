# Voice AI Agent (voice-ai-agent)

Real-time WebSocket voice agent: browser mic → Deepgram STT (`nova-2`) → Gemini (`gemini-2.5-flash`) → Deepgram TTS (`aura-asteria-en`) → browser speaker. Async pipeline with bounded queues.

Files: `agent.py` (404 lines, `CustomVoiceAgent`), `main.py` (47 lines, FastAPI `/ws/voice`), `static/index.html` (154 lines, Web Audio API), `.env` (Deepgram + Gemini keys), `requirements.txt` (fastapi, uvicorn, websockets, httpx, python-dotenv).

## Setup
```bash
cp .env.example .env
# Set DEEPGRAM_API_KEY and GEMINI_API_KEY in .env
pip install -r requirements.txt
```

Note: `.env` currently holds live API keys — add `.env` to `.gitignore` and rotate before any push.

## Run
```bash
# Local
uvicorn main:app --host 0.0.0.0 --port 8000
# Phone (same WiFi): open http://<server-ip>:8000, allow mic
```

WebSocket endpoint: `ws://host/ws/voice?token=<SESSION_TOKEN>` (optional, set `SESSION_TOKEN` in `.env`).

## Pipeline (agent.py)
- `read_client_mic_loop`: sends browser audio to Deepgram STT (WSS `v1/listen`, `interim_results=true`, `endpointing=300`, `vad_events=true`).
- `handle_dg_responses`: receives interim/final results; accumulates transcript; sends to `llm_prompt_queue`. Barge-in guard at ~line 189: only purges if `SpeechStarted` + `is_ai_speaking` + loud ratio (>0.50) — false triggers suppressed.
- `gemini_llm_loop`: streams Gemini SSE; sentence-buffered chunks pushed to `tts_text_queue`. Timeout recovery (`wait_for` 3.0s) at ~line 222.
- `tts_feed_loop`: pulls from `tts_text_queue`; sends to Deepgram TTS (`wss://api.deepgram.com/v1/speak`, `encoding=linear16`, `model=aura-asteria-en`); sends `Flush` after final chunk.
- `write_client_speaker_loop`: writes TTS audio chunks to WebSocket; `is_ai_speaking` decays to False after ~50ms of no new chunk.

Queue bounds prevent backpressure; `_purge_pipeline()` drains only output queues (`tts_text_queue`, audio out) — preserves interrupted user's `audio_in_queue`.

## Interruption / Barge-in
- `interruption_event` triggers purge; `gemini_llm_loop` skips interim `Results` during interruption but lets `speech_final` through.
- Ratio logging at `agent.py:211-213`: `last_rms / 32768.0` printed at interruption.
- False interruption fix applied: loud gate (`ratio > 0.50`) prevents quick-reply false positives; `is_ai_speaking` decay at ~50ms.

## Key Config References
- STT: `nova-2`, 16 kHz linear16, smart_format, endpointing 300 ms
- LLM: `gemini-2.5-flash`, streaming SSE
- TTS: `aura-asteria-en`
- Client audio: `getUserMedia({audio: {echoCancellation: true, noiseSuppression: true}})` in `static/index.html`

## Security / Notes
- `.env` contains real keys (113 bytes). Rotate and `.gitignore`.
- No `ARCHITECTURE.md` in repo; this README replaces explanatory docs.
