"""
FastAPI example for myvoiceai.
This demonstrates how to use `run_voice_session` with a WebSocket endpoint.
"""
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from myvoiceai import run_voice_session
from myvoiceai.tools import ToolRegistry
from fastapi.responses import HTMLResponse
from pathlib import Path
load_dotenv()

GEMINI_API_KEY=os.getenv("GEMINI_API_KEY", "")
DEEPGRAM_API_KEY=os.getenv("DEEPGRAM_API_KEY", "")

app = FastAPI(title="Voice AI Example")

# Allow all CORS for browser clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

@app.get("/")
async def root():
    html_path = BASE_DIR / "static" / "index.html"
    with open(html_path, "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)


@app.websocket("/ws/voice")
async def voice_session(websocket: WebSocket):
    await websocket.accept()
    
    # Build a tool registry (customize as needed)
    my_registry = ToolRegistry()
    
    # Optionally add a custom tool
    async def lookup_flight(**kwargs):
        # This is a placeholder - replace with real logic
        return {
            "flight": "AA123",
            "gate": "B12",
            "departure": kwargs.get("from"),
            "destination": kwargs.get("to"),
        }
    
    my_registry.register(
        name="lookup_flight",
        schema={
            "type": "function",
            "function": {
                "name": "lookup_flight",
                "description": "Look up a flight",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "from": {"type": "string", "description": "Departure airport code"},
                        "to": {"type": "string", "description": "Destination airport code"},
                    },
                    "required": ["from", "to"],
                },
            },
        },
        impl=lookup_flight,
    )
    
    # Example system prompt
    system_prompt = "You are a travel assistant. You can look up flight information using the lookup_flight tool."
    
    # Use the generic greeting
    greeting_message = "Hi there! I'm your travel assistant. "
    
    # Run the voice session with custom tools
    await run_voice_session(
        websocket=websocket,
        system_prompt=system_prompt,
        greeting_message=greeting_message,
        endpointing=1200,
        utterance_end=2500,
        stable_interim_secs=1.5,
        stable_interim_secs_no_punct=3.0,
        inactivity_timeout_seconds=7,
        tool_registry=my_registry,
        llm_provider_api_key=GEMINI_API_KEY,
        deepgram_api_key=DEEPGRAM_API_KEY,
        session_id="test-001",
        tracing=False,
    )


if __name__ == "__main__":
    import uvicorn
    
    print("Starting Voice AI server...")
    print("Access the docs at http://localhost:8000/docs")
    print("Connect WebSocket clients to ws://localhost:8000/ws/voice")
    
    uvicorn.run("example.fastapi_app:app", host="0.0.0.0", port=8000, reload=True)