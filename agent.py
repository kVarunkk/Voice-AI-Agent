import asyncio
import json
import logging
import os
import time
from typing import AsyncIterator, cast
import httpx
import websockets
from websockets.exceptions import ConnectionClosed
from starlette.websockets import WebSocketDisconnect
from litellm import acompletion, exceptions
from utils.strip_markdown import strip_markdown_for_speech

logger = logging.getLogger("voice_agent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s.%(msecs)03d %(levelname)s:%(name)s:%(message)s", datefmt="%H:%M:%S")

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not DEEPGRAM_API_KEY or not GEMINI_API_KEY:
    raise RuntimeError(
        "DEEPGRAM_API_KEY and GEMINI_API_KEY must be set in the environment (.env)"
    )

DEEPGRAM_STT_URL = (
    "wss://api.deepgram.com/v1/listen"
    "?model=nova-2&encoding=linear16&sample_rate=16000&channels=1"
    "&interim_results=true&endpointing=300&smart_format=true"
    "&no_delay=true"
    # "&vad_events=true&utterance_end_ms=1000"
)

DEEPGRAM_TTS_URL = (
    "wss://api.deepgram.com/v1/speak"
    "?encoding=linear16&sample_rate=16000&model=aura-asteria-en"
)
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-2.5-flash:streamGenerateContent?alt=sse&key={GEMINI_API_KEY}"
)
LLM_MODEL = "gemini/gemini-2.5-flash"

SENTENCE_BOUNDARY_CHARS = {".", "!", "?", "\n"}
CLAUSE_BOUNDARY_CHARS = {",", ";", ":"}
MAX_BUFFER_CHARS_BEFORE_FORCED_FLUSH = 100

DEFAULT_MAX_SESSION_SECONDS = 300          
DEFAULT_INACTIVITY_TIMEOUT_SECONDS = 10    
DEFAULT_GOODBYE_WAIT_SECONDS = 10 

SYSTEM_PROMPT = (
    "You are a helpful, concise voice assistant. Keep replies short and "
    "conversational since they will be spoken aloud."
)

BASE_SYSTEM_PROMPT = "Do not use markdown formatting (no asterisks, bullet points, headers, or bold/italic syntax). Your responses are converted to speech, so write in plain spoken sentences only."

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
    def __init__(
        self, 
        client_websocket, 
        system_prompt=SYSTEM_PROMPT, 
        max_session_seconds: float = DEFAULT_MAX_SESSION_SECONDS,
        inactivity_timeout_seconds: float = DEFAULT_INACTIVITY_TIMEOUT_SECONDS,
        max_duration_message: str = "We've reached our time limit for this session, goodbye for now.",
        inactivity_message: str = "I haven't heard from you in a bit, so I'll go ahead and close this session.",
    ):
        self.client_ws = client_websocket

        self.audio_in_queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        self.llm_prompt_queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        self.tts_text_queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        self.audio_out_queue: asyncio.Queue = asyncio.Queue(maxsize=200)

        self.transcript_accumulator = ""
        self.sentence_buffer = ""
        self.is_ai_speaking = False
        self.interruption_event = asyncio.Event()
        self.conversation_history = []  
        self.system_prompt = system_prompt + BASE_SYSTEM_PROMPT

        self._tasks: list[asyncio.Task] = []
        self._dg_stt_ws = None
        self._dg_tts_ws = None

        self._last_audio_sent_ts = None
        self._speech_final_ts = None
        self._gemini_request_ts = None
        self._tts_first_send_ts = None
        self._last_audio_out_ts = None

        self.max_session_seconds = max_session_seconds
        self.inactivity_timeout_seconds = inactivity_timeout_seconds
        self.max_duration_message = max_duration_message
        self.inactivity_message = inactivity_message
        self._last_activity_ts = time.monotonic()

        self._closing = False
        self._session_end_event = asyncio.Event()
        # self._inactivity_start_ts  = time.monotonic()
        self._tts_flush_event = asyncio.Event()


    async def run(self):
        """Orchestrates system loops and guarantees cleanup on exit,
        whether that exit is a client disconnect or an unhandled error
        in any one of the loops."""
        self._tasks = [
            asyncio.create_task(self.read_client_mic_loop(), name="mic_in"),
            asyncio.create_task(self.deepgram_stt_loop(), name="stt"),
            asyncio.create_task(self.llm_loop(), name="llm"),
            asyncio.create_task(self.deepgram_tts_loop(), name="tts"),
            asyncio.create_task(self.write_client_speaker_loop(), name="speaker_out"),
            asyncio.create_task(self.session_timer_loop(), name="session_timer"),
        ]
        try:
            done, pending = await asyncio.wait(
                self._tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            
            for task in done:
                if task.exception():
                    logger.exception(
                        "Task %s failed",
                        task.get_name(),
                        exc_info=task.exception(),
                    )
            
            if self._closing:
                await self._session_end_event.wait()
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
        try:
            await self.client_ws.close()
        except Exception:
            pass        
        logger.info("Session cleaned up.")

# adds audio to the audio_in_queue
    # async def read_client_mic_loop(self):
    #     """Task 1: intercepts raw binary audio chunks from the client."""
    #     try:
    #         while True:
    #             message = await self.client_ws.receive_bytes()
    #             await self.audio_in_queue.put(message)
    #     except WebSocketDisconnect:
    #         logger.info("Client disconnected (mic loop).")          
    # 
    async def read_client_mic_loop(self):
        """Task 1: intercepts raw binary audio chunks from the client,
        and dispatches JSON control/ack messages on the same channel."""
        try:
            while True:
                message = await self.client_ws.receive()
                if message.get("type") == "websocket.disconnect":
                    raise WebSocketDisconnect()

                if "bytes" in message and message["bytes"] is not None:
                    await self.audio_in_queue.put(message["bytes"])
                elif "text" in message and message["text"] is not None:
                    try:
                        data = json.loads(message["text"])
                    except json.JSONDecodeError:
                        continue
                    if data.get("control") == "playback_complete":
                        self._last_activity_ts = time.monotonic()
                        self.is_ai_speaking = False
        except WebSocketDisconnect:
            logger.info("Client disconnected (mic loop).")      

# sends audio to deepgram and receives text via websocket
    async def deepgram_stt_loop(self):
        """Task 2: full-duplex pipe driving live Deepgram STT."""
        self._dg_stt_ws = await _connect_deepgram(DEEPGRAM_STT_URL)
        dg_ws = self._dg_stt_ws
        logger.info("Connected to Deepgram STT.")
    
        async def forward_audio_to_dg():
            while True:
                chunk = await self.audio_in_queue.get()
                await dg_ws.send(chunk)
                self._last_audio_sent_ts = time.monotonic()
                self.audio_in_queue.task_done()
    
        async def dispatch_final_transcript(source: str):
            if not self.transcript_accumulator.strip():
                return
            self._last_activity_ts = time.monotonic()
            final_text = self.transcript_accumulator.strip()
            if self._last_audio_sent_ts:
                logger.info(
                    "STT time (last audio to %s): %.3fs",
                    source, time.monotonic() - self._last_audio_sent_ts
                )
            self._speech_final_ts = time.monotonic()
            logger.info("User: %s", final_text)
            try:
                await self.client_ws.send_json({"transcript": {"role": "user", "text": final_text}})
            except Exception:
                pass
            self.transcript_accumulator = ""
            await self.llm_prompt_queue.put(final_text)
    
        async def handle_dg_responses():
            while True:
                try:
                    msg = await dg_ws.recv()
                    data = json.loads(msg)
                except json.JSONDecodeError:
                    logger.warning("Non-JSON message from Deepgram, skipping.")
                    continue
    
                msg_type = data.get("type")
    
                if msg_type == "UtteranceEnd":
                    logger.info("UtteranceEnd received.")
                    await dispatch_final_transcript("UtteranceEnd")
                    continue
    
                if msg_type == "Results":
                    alt = data.get("channel", {}).get("alternatives", [{}])[0]
                    transcript = alt.get("transcript", "")
    
                    if transcript:
                        self._last_activity_ts = time.monotonic()

                        logger.info("Interim transcript fragment: %r (is_final=%s) (is_speech_final=%d)", transcript, data.get("is_final"), data.get("speech_final"))

                        logger.info("is ai speaking? %d , interruption event set? %s, closing true? %r", self.is_ai_speaking, self.interruption_event.is_set(), self._closing)
    
                        if self.is_ai_speaking and not self.interruption_event.is_set() and not self._closing:
                            logger.info("Barge-in detected via transcript: %r", transcript)
                            self.interruption_event.set()
                            self.transcript_accumulator = ""
                            await self._purge_pipeline()
    
                        if data.get("is_final"):
                            self.transcript_accumulator += f" {transcript}"
    
                    if data.get("speech_final"):
                        await dispatch_final_transcript("speech_final")
    
        try:
            await asyncio.gather(forward_audio_to_dg(), handle_dg_responses())
        except ConnectionClosed:
            logger.warning("Deepgram STT connection closed.")

    async def llm_loop(self):
        """Task 3: streams prompts to the LLM via LiteLLM, splits the response into
        sentence-sized chunks, and forwards each chunk to the TTS queue
        as soon as it is ready to speak."""
        while True:
    
            prompt = await self.llm_prompt_queue.get()

            if self._closing:
                self.llm_prompt_queue.task_done()
                continue
    
            if self._speech_final_ts:
                logger.info("Time from speech end to LLM dispatch: %.3fs", time.monotonic() - self._speech_final_ts)
            self._gemini_request_ts = time.monotonic()
            self._tts_first_send_ts = None
            logger.info("Sending prompt to LLM: %r", prompt)
            self.sentence_buffer = ""
    
            self.conversation_history.append(
                {"role": "user", "content": prompt}
            )
    
            messages = [{"role": "system", "content": self.system_prompt}] + self.conversation_history
    
            full_reply = ""
            first_token_logged = False
            try:
                response = await acompletion(
                    model=LLM_MODEL,
                    messages=messages,
                    stream=True,
                    timeout=30.0,
                )
    
                async for chunk in response:
                    if self.interruption_event.is_set():
                        break
    
                    try:
                        text_token = chunk.choices[0].delta.content
                    except (AttributeError, IndexError):
                        text_token = None
    
                    if not text_token:
                        continue
    
                    full_reply += text_token
                    if not first_token_logged:
                        logger.info("LLM time to first token: %.3fs", time.monotonic() - self._gemini_request_ts)
                        first_token_logged = True
    
                    await self._buffer_and_dispatch(text_token)
    
                if not full_reply.strip():
                    logger.warning(
                        "Empty LLM response for prompt %r (interrupted=%s)",
                        prompt, self.interruption_event.is_set()
                    )
                logger.info("LLM total time to final token: %.3fs", time.monotonic() - self._gemini_request_ts)
    
            except exceptions.APIError:
                logger.exception("LLM request failed.")
            except Exception:
                logger.exception("Unexpected error during LLM streaming.")
    
            # Flush whatever is left in the buffer so the last clause
            # still gets spoken, then tell Deepgram TTS to flush audio.
            if not self._closing:
                if self.sentence_buffer.strip() and not self.interruption_event.is_set():
                    await self.tts_text_queue.put(strip_markdown_for_speech(self.sentence_buffer))
                await self.tts_text_queue.put({"flush": True})
            self.sentence_buffer = ""
    
            if full_reply.strip() and not self.interruption_event.is_set():
                self.conversation_history.append(
                    {"role": "assistant", "content": full_reply}
                )
                try:
                    await self.client_ws.send_json({"transcript": {"role": "assistant", "text": full_reply}})
                except Exception:
                    pass
    
            self.llm_prompt_queue.task_done()

    async def _buffer_and_dispatch(self, token: str):
        self.sentence_buffer += token
    
        flush_idx = -1
        for i, ch in enumerate(self.sentence_buffer):
            if ch in SENTENCE_BOUNDARY_CHARS:
                flush_idx = i
            elif ch in CLAUSE_BOUNDARY_CHARS and i > 40:
                flush_idx = i
    
        if flush_idx != -1:
            text_to_send = strip_markdown_for_speech(self.sentence_buffer[:flush_idx + 1]).strip()
            remainder = self.sentence_buffer[flush_idx + 1:]
            if text_to_send:
                logger.info("Flushing to tts_text_queue at t=%.3f: %r", time.monotonic(), text_to_send)
                await self.tts_text_queue.put(text_to_send)
            self.sentence_buffer = remainder
        elif len(self.sentence_buffer) > MAX_BUFFER_CHARS_BEFORE_FORCED_FLUSH:
            text_to_send = strip_markdown_for_speech(self.sentence_buffer).strip()
            if text_to_send:
                await self.tts_text_queue.put(text_to_send)
            self.sentence_buffer = ""
       

    async def deepgram_tts_loop(self):
        """Task 4: submits buffered text to Deepgram's Aura streaming TTS
        and streams the resulting PCM audio back out."""
        self._dg_tts_ws = await _connect_deepgram(DEEPGRAM_TTS_URL)
        tts_ws = self._dg_tts_ws

        async def feed_text_to_tts():
            while True:
                item = await self.tts_text_queue.get()
                logger.info("feed_text_to_tts dequeued at t=%.3f: %r", time.monotonic(), item)
                if isinstance(item, dict) and item.get("flush"):
                    logger.info("Sending Flush to Deepgram TTS.")
                    await tts_ws.send(json.dumps({"type": "Flush"}))
                elif not self.interruption_event.is_set():
                    if self._tts_first_send_ts is None:
                        self._tts_first_send_ts = time.monotonic()
                    logger.info("Sending text to Deepgram TTS at t=%.3f: %r", time.monotonic(), item)
                    await tts_ws.send(json.dumps({"type": "Speak", "text": item}))
                self.tts_text_queue.task_done()

        async def harvest_audio_from_tts():
            while True:
                msg = await tts_ws.recv()
                if isinstance(msg, bytes) and not self.interruption_event.is_set():
                    if self._tts_first_send_ts is not None:
                        logger.info("TTS time to first audio: %.3fs", time.monotonic() - self._tts_first_send_ts)
                        self._tts_first_send_ts = None
                    await self.audio_out_queue.put(msg)
                else:
                    try:
                        data = json.loads(msg)
                
                        if data.get("type") == "Flushed":
                            self._tts_flush_event.set()
                
                    except json.JSONDecodeError:
                        pass    

        try:
            await asyncio.gather(feed_text_to_tts(), harvest_audio_from_tts())
        except ConnectionClosed:
            logger.warning("Deepgram TTS connection closed.")

    async def write_client_speaker_loop(self):
        while True:
            audio_payload = await self.audio_out_queue.get()
            self._last_audio_out_ts = time.monotonic()
            # self._last_activity_ts = self._last_audio_out_ts
            if not self.interruption_event.is_set():
                self.is_ai_speaking = True
                try:
                    await self.client_ws.send_bytes(audio_payload)
                except WebSocketDisconnect:
                    logger.info("Client disconnected (speaker loop).")
                    return
            self.audio_out_queue.task_done()

    async def session_timer_loop(self):
        """Task 6: enforces a max session duration and an inactivity timeout,
        closing the session with a spoken message when either is hit."""
        session_start = time.monotonic()
        while True:
            await asyncio.sleep(1)
    
            now = time.monotonic()
    
            if now - session_start > self.max_session_seconds:
                logger.info("Max session duration (%.0fs) reached, closing.", self.max_session_seconds)
                await self._say_goodbye_and_close(self.max_duration_message)
                return
    
            if (now - self._last_activity_ts > self.inactivity_timeout_seconds) and not self.is_ai_speaking:
                logger.info("Inactivity timeout (%.0fs) reached, closing.", self.inactivity_timeout_seconds)
                await self._say_goodbye_and_close(self.inactivity_message)
                return

    async def _say_goodbye_and_close(self, text: str):
        """Halts any in-flight LLM/TTS output, then speaks a final message
        through the pipeline and waits for it to finish playing."""
        self._closing = True
        self._tts_flush_event.clear()
        logger.info("Executing goodbye sequence: %r", text)

        self.interruption_event.set()
        await self._purge_pipeline()

        # 3. Ensure state triggers let the text pass through cleanly
        self.interruption_event.clear() 
        self.is_ai_speaking = True  # Hold this True so the loop knows audio is expected
    
        # 4. Enqueue the final text
        await self.tts_text_queue.put(strip_markdown_for_speech(text))
        await self.tts_text_queue.put({"flush": True})
        
        # 5. Wait for the audio to generate, land in the queue, and finish playing
        # Wait a brief moment for the generator to catch up before checking if queues are empty
        await asyncio.sleep(0.3) 
        
        try:
            await asyncio.wait_for(
                self._tts_flush_event.wait(),
                timeout=DEFAULT_GOODBYE_WAIT_SECONDS,
            )
            logger.info("Deepgram finished generating goodbye audio.")
        except asyncio.TimeoutError:
            logger.warning("Timed out waiting for Deepgram goodbye audio.")

        self._session_end_event.set()    
    

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
       