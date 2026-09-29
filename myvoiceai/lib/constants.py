import os

OTEL_EXPORTER_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces")
# DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPGRAM_STT_MODEL="nova-2"
DEEPGRAM_TTS_MODEL="aura-asteria-en"
DEEPGRAM_STT_URL = (
    "wss://api.deepgram.com/v1/listen"
    "?model={stt_model}&encoding=linear16&sample_rate=16000&channels=1"
    "&interim_results=true&endpointing={endpointing}&smart_format=true"
    "&no_delay=true"
    "&vad_events=true"
    "&utterance_end_ms={utterance_end}"
)

DEEPGRAM_TTS_URL = (
    "wss://api.deepgram.com/v1/speak"
    "?encoding=linear16&sample_rate=16000&model={tts_model}"
)
LLM_MODEL = "gemini/gemini-2.5-flash"

SENTENCE_BOUNDARY_CHARS = {".", "!", "?", "\n"}
CLAUSE_BOUNDARY_CHARS = {",", ";", ":"}
MAX_BUFFER_CHARS_BEFORE_FORCED_FLUSH = 100

DEFAULT_MAX_SESSION_SECONDS = 300          
DEFAULT_INACTIVITY_TIMEOUT_SECONDS = 10    
DEFAULT_GOODBYE_WAIT_SECONDS = 10 

DEFAULT_GREETING_MESSAGE = "Hi there! How can I help you today?"

SYSTEM_PROMPT = (
    "You are a helpful, concise voice assistant. Keep replies short and "
    "conversational since they will be spoken aloud."
)

BASE_SYSTEM_PROMPT = "Do not use markdown formatting (no asterisks, bullet points, headers, or bold/italic syntax). Your responses are converted to speech, so write in plain spoken sentences only."

GUARDRAIL_BLOCK_MESSAGE = "I can't help with that."
MAX_DURATION_MESSAGE="We've reached our time limit for this session, goodbye for now."
INACTIVITY_MESSAGE="I haven't heard from you in a bit, so I'll go ahead and close this session."
TOOL_FILLER_PHRASES = [
    "Let me check on that.",
    "One sec, looking into it.",
    "Give me a moment.",
]
TOOL_CALL_TIMEOUT_SECONDS = 8.0
MAX_TOOL_HOPS = 3

DEFAULT_ENDPOINTING = 1200
DEFAULT_UTTERANCE_END = 2500
STABLE_INTERIM_SECS = 1.5
STABLE_INTERIM_NO_PUNCT_SECS = 3.0

FILLERS = {"and", "uh", "um", "so", "but", "or", "the", "a", "like"}