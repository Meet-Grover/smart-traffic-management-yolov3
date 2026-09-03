"""FastAPI app: upload a frame per lane, get live signal decisions back."""
from __future__ import annotations

import asyncio
import io

import numpy as np
import cv2
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.detection import detect_frame
from app.signal_control import SignalController

LANE_NAMES = ["North", "East", "South", "West"]

app = FastAPI(title="Smart Traffic Management System")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

controller = SignalController(LANE_NAMES)

_websockets: list[WebSocket] = []


def _decode_upload(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Uploaded file is not a readable image")
    return frame


@app.get("/api/state")
def get_state():
    return controller.state.as_dict()


@app.post("/api/lanes/{lane_id}/detect")
async def detect_lane(lane_id: int, file: UploadFile = File(...)):
    if not 0 <= lane_id < len(LANE_NAMES):
        return {"error": "invalid lane_id"}, 400

    data = await file.read()
    frame = _decode_upload(data)
    result = detect_frame(frame)

    counts = [l.vehicle_count for l in controller.state.lanes]
    emergencies = [l.emergency for l in controller.state.lanes]
    counts[lane_id] = result.vehicle_count
    emergencies[lane_id] = result.emergency_detected

    state = controller.update(counts, emergencies)
    await _broadcast(state.as_dict())

    return {
        "lane": LANE_NAMES[lane_id],
        "vehicle_count": result.vehicle_count,
        "emergency_detected": result.emergency_detected,
        "signal_state": state.as_dict(),
    }


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    _websockets.append(websocket)
    try:
        await websocket.send_json(controller.state.as_dict())
        while True:
            await websocket.receive_text()  # keep-alive; client doesn't need to send anything meaningful
    except WebSocketDisconnect:
        _websockets.remove(websocket)


async def _broadcast(payload: dict):
    dead = []
    for ws in _websockets:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _websockets.remove(ws)


# Serve the frontend (index.html + samples/) from this same server, at
# the very end so it never shadows the API routes registered above.
# This also gives the page a real http:// origin instead of file://,
# which fixes canvas/WebSocket restrictions Safari and Chrome enforce
# on local files.
_frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
if _frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
