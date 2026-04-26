# AI-Based Real-Time Surgical Tool Tracking with Dynamic Danger Zone Alerting

A modular, production-quality computer-vision system that detects and tracks
surgical instruments in live video, highlights configurable danger zones, and
triggers immediate visual (and optional audio) alerts when a tool enters a
restricted area.

---

## Features

| Feature | Details |
|---|---|
| **Object Detection** | YOLOv8 (nano by default — swappable for any YOLO variant or custom model) |
| **Multi-Object Tracking** | DeepSORT — persistent IDs across frames |
| **Danger Zones** | Multiple configurable rectangular zones per scene |
| **Alert System** | Red bounding box, flashing banner, console log, optional Windows beep |
| **FPS Counter** | Rolling 30-frame average shown on screen |
| **Pause / Resume** | Press `P` while the window is focused |
| **Simulation mode** | Works with any object (no custom surgical dataset needed) |

---

## Project Structure

```
realtime-surgical-tool-tracking-ai/
├── main.py                    ← Entry point
├── requirements.txt
├── README.md
│
├── config/
│   └── config.yaml            ← All tunable parameters
│
├── detection/
│   ├── __init__.py
│   └── detector.py            ← YOLOv8 wrapper → List[Detection]
│
├── tracking/
│   ├── __init__.py
│   └── tracker.py             ← DeepSORT wrapper → List[Track]
│
└── utils/
    ├── __init__.py
    ├── danger_zone.py         ← Zone definitions + collision logic
    └── visualization.py       ← All OpenCV drawing code
```

---

## Setup

### 1. Prerequisites

- Python 3.10 or higher
- A webcam **or** a video file

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> On first run, `ultralytics` will automatically download `yolov8n.pt`
> (~6 MB) if it is not already present.

---

## Running the System

### Default (webcam)

```bash
python main.py
```

### Video file

```bash
python main.py --source path/to/video.mp4
```

### Custom config

```bash
python main.py --config path/to/my_config.yaml
```

### All options

```
python main.py --help

  --config  CONFIG   Path to config.yaml (default: config/config.yaml)
  --source  SOURCE   Webcam index (0,1,…) or path to video file
```

---

## Keyboard Controls

| Key | Action |
|-----|--------|
| `P` | Pause / resume |
| `Q` | Quit |

---

## Configuration (`config/config.yaml`)

```yaml
model:
  path: "yolov8n.pt"     # Any YOLOv8 weights file or custom model
  confidence: 0.45        # Detection threshold
  device: "cpu"           # "cpu" | "cuda" | "mps"

video:
  source: 0               # Webcam index or video file path
  width:  1280
  height: 720

danger_zones:
  - name: "Critical Zone A"
    coordinates: [50, 50, 420, 370]    # [x1, y1, x2, y2] pixels
    color: [0, 0, 255]                  # BGR

surgical_tools:
  simulate: true          # true = label ALL objects as tools

alerts:
  console_log: true
  sound: false            # Windows beep — set true to enable
  overlap_threshold: 0.0  # 0 = any touch triggers alert
```

### Tuning danger zones to your resolution

1. Run the system to see your video at its actual resolution (shown in the
   terminal as `resolution: WxH`).
2. Edit `coordinates` in `config.yaml` to match your scene.
3. Restart — no code changes needed.

---

## Using a Custom Surgical-Tool Model

1. Train or download a YOLO model for surgical instruments (e.g. from
   [Roboflow Universe](https://universe.roboflow.com/) — search "surgical tools").
2. Set `model.path` in `config.yaml` to the `.pt` file path.
3. Set `surgical_tools.simulate: false` and list your class IDs under
   `surgical_tools.classes`.

---

## Example Output

```
============================================================
  Surgical Tool Tracking — Initialising
============================================================
[Detector] Loading model 'yolov8n.pt' on device='cpu' ...
[Detector] Ready  (conf=0.45, simulate=True)
[Tracker]  DeepSORT initialised.
[DangerZone] Loaded 'Critical Zone A'  coords=[50, 50, 420, 370]
[DangerZone] Loaded 'Restricted Zone B' coords=[620, 160, 980, 460]

[INFO] Video source opened — resolution: 1280×720
[INFO] Press  P  to pause/resume   |   Q  to quit

[ALERT] Tool ID=1 entered 'Critical Zone A'
[ALERT] Tool ID=3 entered 'Restricted Zone B'
```

On screen you will see:
- Semi-transparent coloured rectangles marking each danger zone
- Green bounding boxes with `ID:N  tool:<class>` labels for safe tools
- **Red** bounding boxes with `⚠ DANGER` for tools inside a zone
- A flashing red banner at the bottom listing all active violations
- FPS counter in the top-right corner

---

## Future Improvements

- [ ] Mouse-drawn interactive zone editor (click-and-drag in the window)
- [ ] Custom YOLOv8 model fine-tuned on real surgical instrument datasets
- [ ] Trajectory / heatmap visualisation showing tool movement history
- [ ] Export alerts to a timestamped log file (CSV / JSON)
- [ ] Network streaming output (RTSP / WebRTC)
- [ ] REST API for zone management at runtime
- [ ] GPU acceleration (set `device: cuda` in config)
- [ ] Integration with robotic-arm safety stop signal

---

## License

MIT — free to use, modify, and distribute.
