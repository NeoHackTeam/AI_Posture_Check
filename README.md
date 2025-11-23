# 🧍‍♂️ PosturePal  
### **AI-Powered Posture Detection, Break Timer & Productivity Assistant**

PosturePal is an intelligent posture-monitoring system combining **computer vision**, **pose detection**, **AI chat**, and **productivity tools** to help users maintain healthy posture while working.

It includes:

1. **A Web-Based PosturePal Interface (Flask + HTML)**  
2. **A Standalone Desktop App (OpenCV + MediaPipe)**  

Both systems are included in this repository.

---

# 📂 Project Structure

```
PosturePal/
│
├── chatbot.py                 # Flask server for website + chatbot (port 5000)
├── posture_server.py          # Flask webcam posture server (port 8000)
├── cameraposture.py           # Standalone AI posture detection app
│
├── index.html                 # Web UI interface
├── pose_landmarker_full.task  # MediaPipe pose model
├── posture_settings.json      # User settings + calibration data
├── posture_log.csv            # Session logs
│
└── README.md                  # Documentation
```

---

# 🌐 1. Web PosturePal (Flask + HTML)

The web interface allows users to:

- Interact with the built-in AI chatbot  
- Launch the camera demo for posture detection  
- Access instructions, UI panels, and modals  

### Architecture

| Component | File | Port | Purpose |
|----------|------|------|---------|
| Main Web Server | `chatbot.py` | 5000 | Serves UI + chatbot |
| Camera Server | `posture_server.py` | 8000 | Handles webcam posture detection |
| Frontend | `index.html` | — | UI for PosturePal |

### Flow

```
index.html
   │
   ├── fetch("http://localhost:5000/...")   # chatbot responses
   └── window.open("http://localhost:8000") # posture camera page
```

Both servers **must** run at the same time for full functionality.

---

# ▶️ How to Run the Web Version

### Terminal 1 — Start main UI server
```bash
python chatbot.py
```

Open in browser:
```
http://localhost:5000
```

### Terminal 2 — Start webcam posture server
```bash
python posture_server.py
```

Then click **Start Web Demo** in the UI.

---

# 🧠 2. Standalone PosturePal Desktop App

The desktop application performs full real-time posture detection with OpenCV + MediaPipe and includes a Pomodoro-style work/break timer.

Run with:
```bash
python cameraposture.py
```

---

# 📸 Real-Time Pose Detection

PosturePal uses MediaPipe PoseLandmarker to detect:

- Head + neck points  
- Shoulders  
- Elbows  
- Hips  
- Knees  

Runs at **25–30 FPS**.

---

# 📐 Angle Calculations

Three major posture angles are computed:

- **Neck angle** → forward head posture  
- **Shoulder angle** → rounded shoulders  
- **Back angle** → slouching  

Angle formula:

```
angle = arccos((ba · bc) / (|ba| × |bc|))
```

Angles are smoothed using an **8-frame median filter**.

---

# 🔔 Smart Notifications

Notifications trigger when:

- Bad posture persists for **10+ seconds**  
- Includes Windows toast notification  
- Loud audio alert  
- Console log entry  

PosturePal waits for posture to improve before re-triggering alerts.

---

# ⏱️ Work/Break Timer

Features:

- Default: **30s work, 5s break**  
- Break screen with stretch prompts  
- Timer widget  
- Session counter  
- Pause/Resume  

---

# 🎯 Calibration

Press `C` to calibrate:

1. Sit straight for ~2 seconds  
2. Captures ~60 frames  
3. Calculates your baseline angles  
4. Adjusts thresholds  
5. Saves to `posture_settings.json`

Press `S` to skip.

---

# 🖥️ UI Features

- Skeleton overlay (green = good, red = bad)  
- Neck/shoulder/back angle readouts  
- Smoothed vs. raw angles  
- Posture status panel  
- Timer widget  
- Break screen  
- Keyboard help overlay  

---

# ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `C` | Calibrate |
| `S` | Skip calibration |
| `SPACE` | Pause/resume timer |
| `Q` | Quit |
| `H` | Toggle help overlay |

---

# ⚙️ posture_settings.json Example

```json
{
  "work_interval": 30,
  "break_duration": 5,
  "neck_angle_threshold": 155,
  "shoulder_slouch_threshold": 145,
  "back_angle_threshold": 145,
  "posture_warning_duration": 10,
  "face_distance_threshold": 0.25,
  "model_path": "pose_landmarker_full.task",
  "show_landmarks": true,
  "minimal_mode": false,
  "sound_enabled": true,
  "enable_calibration": true
}
```

---

# 🔧 Troubleshooting

### Camera Not Opening
- Run both Flask servers in **separate terminals**  
- Check webcam permissions  
- Ensure no other app is using the camera  

### Port 5000 or 8000 Not Working
- One server may not be running  
- Restart both  

### “Model Not Found”
Make sure:
```
pose_landmarker_full.task
```
is in the project directory.

### No Notifications
- Install `win10toast`  
- Enable Windows notifications  
- Hold bad posture for 10+ seconds  

---

# 🧱 System Architecture Overview

```
 ┌────────────────────────────────────────────┐
 │                 index.html                  │
 │          (Web UI + Chat Interface)          │
 └────────────────────────────────────────────┘
                 │                 │
                 ▼                 ▼
 ┌───────────────────────┐   ┌──────────────────────┐
 │ Flask Server : 5000    │   │ Flask Server : 8000   │
 │   chatbot.py           │   │   posture_server.py   │
 └───────────────────────┘   └──────────────────────┘
                 │
                 ▼
 ┌────────────────────────────────────────────┐
 │    Desktop Posture Engine (OpenCV + MP)    │
 │      cameraposture.py / posture_pal.py      │
 └────────────────────────────────────────────┘
```

---

# 🚀 Future Improvements

- Web-based pose detection (WebAssembly)  
- Animated stretch routines  
- Weekly analytics dashboard  
- AI posture coaching  
- Multi-user support  
- Dark mode  

---

# 🔐 Privacy

- 100% local  
- No cloud storage  
- Webcam frames never saved  
- All processing happens on-device  

---

# 📄 License & Attribution

Uses:

- Google MediaPipe  
- OpenCV  
- NumPy  
- Flask  
- win10toast  

---

# 🎉 Thank You for Using PosturePal!
