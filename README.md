# Smart Traffic Management System

Adaptive traffic-signal control from live vehicle detection: each lane's
camera feed is analyzed with YOLOv8, and the intersection's signal timing
adapts to real vehicle density — with emergency vehicles able to pre-empt
the signal in their favor.

This is a refactor of an earlier prototype. The original used YOLOv3 via
raw OpenCV `dnn`, a Tkinter desktop UI, an external paid Azure Vision API
call for ambulance detection, a live MapMyIndia routing dependency, and
persisted its "signal state" as a global array pickled to disk on every
read/write. All of that has been replaced below.

## Architecture

- **`backend/`** — FastAPI service.
  - `app/detection.py` — YOLOv8 (Ultralytics) vehicle detection. Runs
    fully locally, no external API calls or keys required.
  - `app/signal_control.py` — the adaptive signal logic, as an explicit,
    testable state machine (no global pickle files).
  - `app/main.py` — REST + WebSocket API.
- **`frontend/index.html`** — a single-page live dashboard (no build step
  needed) showing all 4 lanes, current signal state, and detection
  results.
- **`legacy/`** — the original implementation, kept for reference only.

## Honest limitation

YOLOv8's pretrained weights come from the COCO dataset, which has no
"ambulance" class. Rather than claim false accuracy, emergency-vehicle
detection here is a documented heuristic (large "truck"/"bus" boxes with a
red/blue light-bar color signature near the roof) — noticeably weaker than
a purpose-trained model. The correct next step, if pursued further, is to
fine-tune YOLOv8 on a labeled emergency-vehicle dataset and swap in those
weights at the single point marked in `detection.py`.

## Running it

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Then open `frontend/index.html` in a browser (or serve it with any static
file server) and upload a traffic image/frame for each of the four lanes.
The dashboard updates live via WebSocket as detections come in.

## API

- `GET /api/state` — current signal state for all lanes.
- `POST /api/lanes/{0-3}/detect` — upload an image (`file`) for a lane;
  runs detection, updates the signal controller, returns the result.
- `WS /ws` — pushes the live signal state on every update.
