import asyncio
import json
import logging
import os
import time
import httpx
import websockets
from websockets.exceptions import ConnectionClosed
from difflib import SequenceMatcher

logger = logging.getLogger("voice_agent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s.%(msecs)03d %(levelname)s:%(name)s:%(message)s", datefmt="%H:%M:%S")

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not DEEPGRAM_API_KEY or not GEMINI_API_KEY:
    raise RuntimeError(
        "DEEPGRAM_API_KEY and GEMINI_API_KEY must be set in the environment (.env)"
    )

# DEEPGRAM_STT_URL = (
#     "wss://api.deepgram.com/v1/listen"
#     "?model=nova-2&encoding=linear16&sample_rate=16000&channels=1"
#     "&interim_results=true&endpointing=300&smart_format=true"
# )
DEEPGRAM_STT_URL = (
    "wss://api.deepgram.com/v1/listen"
    "?model=nova-2&encoding=linear16&sample_rate=16000&channels=1"
    "&interim_results=true&endpointing=300&smart_format=true"
    "&vad_events=true"
)
DEEPGRAM_TTS_URL = (
    "wss://api.deepgram.com/v1/speak"
    "?encoding=linear16&sample_rate=16000&model=aura-asteria-en"
)
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-2.5-flash:streamGenerateContent?alt=sse&key={GEMINI_API_KEY}"
)

SENTENCE_BOUNDARY_CHARS = {".", "!", "?", "\n"}
CLAUSE_BOUNDARY_CHARS = {",", ";", ":"}
MAX_BUFFER_CHARS_BEFORE_FORCED_FLUSH = 200

SYSTEM_PROMPT = (
    "You are a helpful, concise voice assistant. Keep replies short and "
    "conversational since they will be spoken aloud."
)

BARGE_IN_RMS_RATIO = 0.02
MAX_AMP=32768.0

async def _connect_deepgram(url: str):
    """Connect to a Deepgram websocket, tolerating both old and new
    versions of the `websockets` library (the auth-header kwarg was
    renamed from extra_headers to additional_headers in v14)."""
    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
    try:
        return await websockets.connect(url, additional_headers=headers)
    except TypeError:
        return await websockets.connect(url, extra_headers=headers)


class CustomVoiceAgent:
    def __init__(self, client_websocket):
        self.client_ws = client_websocket

        # Bounded queue bridges. Bounded so a stalled consumer applies
        # backpressure instead of letting memory grow without limit.
        self.audio_in_queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        # Audio threshold: chunks with RMS below this percentage of max
        # amplitude are ignored (filters noise/echo/breath). Default 2%.
        self.audio_threshold_ratio = 0.001
        self.llm_prompt_queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        self.tts_text_queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        self.audio_out_queue: asyncio.Queue = asyncio.Queue(maxsize=200)

        # Core state
        self.transcript_accumulator = ""
        self.sentence_buffer = ""
        self.is_ai_speaking = False
        self.interruption_event = asyncio.Event()
        self.conversation_history = []  # list of {"role", "parts"}

        self._tasks: list[asyncio.Task] = []
        self._dg_stt_ws = None
        self._dg_tts_ws = None

        self._last_audio_sent_ts = None
        self._speech_final_ts = None
        self._gemini_request_ts = None
        self._tts_first_send_ts = None
        self._last_audio_out_ts = None


    async def run(self):
        """Orchestrates system loops and guarantees cleanup on exit,
        whether that exit is a client disconnect or an unhandled error
        in any one of the loops."""
        self._tasks = [
            asyncio.create_task(self.read_client_mic_loop(), name="mic_in"),
            asyncio.create_task(self.deepgram_stt_loop(), name="stt"),
            asyncio.create_task(self.gemini_llm_loop(), name="llm"),
            asyncio.create_task(self.deepgram_tts_loop(), name="tts"),
            asyncio.create_task(self.write_client_speaker_loop(), name="speaker_out"),
        ]
        try:
            # Only shut down when an actual exception kills a loop.
            # Normal exits (e.g., Deepgram websocket close) must not
            # kill the whole agent — that was the "stops after barge-in"
            # root cause.
            done, pending = await asyncio.wait(
                self._tasks, return_when=asyncio.FIRST_EXCEPTION
            )
            for task in done:
                if task.exception():
                    logger.exception("Task %s failed", task.get_name(), exc_info=task.exception())
        finally:
            await self._shutdown()

    async def _shutdown(self):
        for task in self._tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

        for ws in (self._dg_stt_ws, self._dg_tts_ws):
            if ws is not None:
                try:
                    await ws.close()
                except Exception:
                    pass
        logger.info("Session cleaned up.")

# adds audio to the audio_in_queue
    async def read_client_mic_loop(self):
        """Task 1: intercepts raw binary audio chunks from the client."""
        while True:
            message = await self.client_ws.receive_bytes()
            await self.audio_in_queue.put(message)

# sends audio to deepgram and receives text via websocket
    async def deepgram_stt_loop(self):
        """Task 2: full-duplex pipe driving live Deepgram STT."""
        self._dg_stt_ws = await _connect_deepgram(DEEPGRAM_STT_URL)
        dg_ws = self._dg_stt_ws
        logger.info("Connected to Deepgram STT.")

# get audio chunks from the audio_in_queue and forward to deepgram
        async def forward_audio_to_dg():
            while True:
                chunk = await self.audio_in_queue.get()
                await dg_ws.send(chunk)
                self._last_audio_sent_ts = time.monotonic()
                self.audio_in_queue.task_done()


        async def handle_dg_responses():
            while True:
                try:
                    msg = await dg_ws.recv()
                    data = json.loads(msg)
                except json.JSONDecodeError:
                    logger.warning("Non-JSON message from Deepgram, skipping.")
                    continue
        
                msg_type = data.get("type")
        
                if msg_type == "Results":
                    alt = data.get("channel", {}).get("alternatives", [{}])[0]
                    transcript = alt.get("transcript", "")

                    if transcript:
                        logger.info("Interim transcript fragment: %r (is_final=%s) (is_speech_final=%d)", transcript, data.get("is_final"), data.get("speech_final"))

# looks for the first word the user speaks => clears transcript => purges pipeline => data.get("is_final") & data.get("speech_final") evaluates to false => next iteration of the loop starts
                        if self.is_ai_speaking and not self.interruption_event.is_set():
                            logger.info("Barge-in detected via transcript: %r", transcript)
                            self.interruption_event.set()
                            self.transcript_accumulator = ""
                            await self._purge_pipeline()

                        if data.get("is_final"):
                            self.transcript_accumulator += f" {transcript}"

                    if data.get("speech_final") and self.transcript_accumulator.strip():
                        final_text = self.transcript_accumulator.strip()
                        if self._last_audio_sent_ts:
                            logger.info("STT time (last audio to speech_final): %.3fs", time.monotonic() - self._last_audio_sent_ts)
                        self._speech_final_ts = time.monotonic()
                        logger.info("User: %s", final_text)
                        self.transcript_accumulator = ""
                        await self.llm_prompt_queue.put(final_text)

        try:
            await asyncio.gather(forward_audio_to_dg(), handle_dg_responses())
        except ConnectionClosed:
            logger.warning("Deepgram STT connection closed.")

    async def gemini_llm_loop(self):
        """Task 3: streams prompts to Gemini, splits the response into
        sentence-sized chunks, and forwards each chunk to the TTS queue
        as soon as it is ready to speak."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            while True:
                try:
                    prompt = await asyncio.wait_for(
                        self.llm_prompt_queue.get(), timeout=3.0
                    )
                except asyncio.TimeoutError:
                    # False interruption: no new user input arrived.
                    # Auto-clear so pipeline doesn't deadlock.
                    if self.interruption_event.is_set():
                        logger.info("No new turn after interruption; clearing event.")
                        self.interruption_event.clear()
                        self.is_ai_speaking = False
                        continue
                    else:
                        continue
               
                if self._speech_final_ts:
                    logger.info("Time from speech end to Gemini dispatch: %.3fs", time.monotonic() - self._speech_final_ts)
                self._gemini_request_ts = time.monotonic()
                self._tts_first_send_ts = None 
                logger.info("Sending prompt to Gemini: %r", prompt)
                self.sentence_buffer = ""

                self.conversation_history.append(
                    {"role": "user", "parts": [{"text": prompt}]}
                )

                payload = {
                    "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                    "contents": self.conversation_history,
                }

                full_reply = ""
                first_token_logged = False
                try:
                    async with client.stream("POST", GEMINI_URL, json=payload) as response:
                        logger.info("Gemini response status: %s", response.status_code)
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if self.interruption_event.is_set():
                                break
                            decoded = line.strip()
                            if not decoded.startswith("data:"):
                                continue
                            data_str = decoded[len("data:"):].strip()
                            if not data_str or data_str == "[DONE]":
                                continue
                            try:
                                chunk = json.loads(data_str)
                                text_token = (
                                    chunk["candidates"][0]["content"]["parts"][0]["text"]
                                )
                            except (KeyError, IndexError, json.JSONDecodeError) as e:
                                logger.warning("Unparseable Gemini chunk: %r (%s)", data_str, e)
                                continue

                            full_reply += text_token
                            if not first_token_logged:
                                self.interruption_event.clear()
                                logger.info("Gemini time to first token: %.3fs", time.monotonic() - self._gemini_request_ts)
                                first_token_logged = True
                            await self._buffer_and_dispatch(text_token)

                        if not full_reply.strip():
                            logger.warning(
                                "Empty Gemini response for prompt %r (interrupted=%s)",
                                prompt, self.interruption_event.is_set()
                            )
                        logger.info("Gemini total time to final token: %.3fs", time.monotonic() - self._gemini_request_ts)

                            
                except httpx.HTTPError:
                    logger.exception("Gemini request failed.")

                # Flush whatever is left in the buffer so the last clause
                # still gets spoken, then tell Deepgram TTS to flush audio.
                if self.sentence_buffer.strip() and not self.interruption_event.is_set():
                    await self.tts_text_queue.put(self.sentence_buffer)
                await self.tts_text_queue.put({"flush": True})
                self.sentence_buffer = ""

                if full_reply.strip() and not self.interruption_event.is_set():
                    self.conversation_history.append(
                        {"role": "model", "parts": [{"text": full_reply}]}
                    )
                self.llm_prompt_queue.task_done()

    async def _buffer_and_dispatch(self, token: str):
        """Accumulates streamed tokens and releases them to TTS at
        sentence or clause boundaries, so Deepgram Aura gets coherent
        chunks of text instead of single tokens."""
        self.sentence_buffer += token
        last_char = token[-1] if token else ""

        should_flush = (
            last_char in SENTENCE_BOUNDARY_CHARS
            or (last_char in CLAUSE_BOUNDARY_CHARS and len(self.sentence_buffer) > 40)
            or len(self.sentence_buffer) > MAX_BUFFER_CHARS_BEFORE_FORCED_FLUSH
        )
        if should_flush:
            await self.tts_text_queue.put(self.sentence_buffer)
            self.sentence_buffer = ""

    async def deepgram_tts_loop(self):
        """Task 4: submits buffered text to Deepgram's Aura streaming TTS
        and streams the resulting PCM audio back out."""
        self._dg_tts_ws = await _connect_deepgram(DEEPGRAM_TTS_URL)
        tts_ws = self._dg_tts_ws

        async def feed_text_to_tts():
            while True:
                item = await self.tts_text_queue.get()
                if isinstance(item, dict) and item.get("flush"):
                    logger.info("Sending Flush to Deepgram TTS.")
                    await tts_ws.send(json.dumps({"type": "Flush"}))
                elif not self.interruption_event.is_set():
                    if self._tts_first_send_ts is None:
                        self._tts_first_send_ts = time.monotonic()
                    logger.info("Sending text to Deepgram TTS: %r", item)
                    await tts_ws.send(json.dumps({"type": "Speak", "text": item}))
                self.tts_text_queue.task_done()

        async def harvest_audio_from_tts():
            while True:
                msg = await tts_ws.recv()
                if isinstance(msg, bytes) and not self.interruption_event.is_set():
                    # logger.info("Received %d bytes of audio from Deepgram TTS.", len(msg))
                    if self._tts_first_send_ts is not None:
                        logger.info("TTS time to first audio: %.3fs", time.monotonic() - self._tts_first_send_ts)
                        self._tts_first_send_ts = None
                    await self.audio_out_queue.put(msg)

        try:
            await asyncio.gather(feed_text_to_tts(), harvest_audio_from_tts())
        except ConnectionClosed:
            logger.warning("Deepgram TTS connection closed.")

    async def write_client_speaker_loop(self):
        # Keep is_ai_speaking stable: stay True while audio is playing or
        # TTS is feeding, plus a short decay so barge-in triggers.
        while True:
            audio_payload = await self.audio_out_queue.get()
            self._last_audio_out_ts = time.monotonic()
            if not self.interruption_event.is_set():
                self.is_ai_speaking = True
                await self.client_ws.send_bytes(audio_payload)
            self.audio_out_queue.task_done()
            # Only drop the flag once both pipelines are really quiet.
            # Use near-zero decay so False barge-in from finished AI
            # doesn't keep is_ai_speaking alive.
            decay = (self._last_audio_out_ts is None or
                     (time.monotonic() - self._last_audio_out_ts) > 0.1)
            if (self.audio_out_queue.empty() and self.tts_text_queue.empty() and decay and not self.interruption_event.is_set()):
                self.is_ai_speaking = False

    async def _purge_pipeline(self):
        """Drains in-flight queues instantly during a user barge-in and
        tells the browser to drop whatever it has already buffered."""
        # Purge AI output only; interrupted user audio (audio_in_queue)
        # must stay alive so the interrupted turn becomes the new prompt.
        for q in (self.llm_prompt_queue, self.tts_text_queue, self.audio_out_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                    q.task_done()
                except asyncio.QueueEmpty:
                    break
        self.is_ai_speaking = False
        self.interruption_event.clear()
        try:
            await self.client_ws.send_json({"control": "clear_speaker_buffer"})
        except Exception:
            pass
        if self._dg_tts_ws is not None:
            try:
                await self._dg_tts_ws.send(json.dumps({"type": "Clear"}))
            except Exception:
                pass