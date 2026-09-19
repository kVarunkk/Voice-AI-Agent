import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()


from agent import CustomVoiceAgent


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice_agent.main")

# A shared secret the client must send as a query param, e.g.
# ws://host/ws/voice?token=xyz. Fine for a single-tenant demo; swap for
# real session auth (JWT, signed cookie, etc.) before going further.
SESSION_TOKEN = os.getenv("SESSION_TOKEN")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)


@app.websocket("/ws/voice")
async def ws_voice_agent_handler(websocket: WebSocket):
    if SESSION_TOKEN and websocket.query_params.get("token") != SESSION_TOKEN:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()
    session_agent = CustomVoiceAgent(websocket)

    try:
        await session_agent.run()
    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception:
        logger.exception("Unhandled error in voice session.")
