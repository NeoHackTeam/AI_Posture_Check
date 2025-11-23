🧍‍♂️ PosturePal
AI-Powered Posture Detection, Break Timer & Productivity Assistant

PosturePal is an intelligent posture-monitoring system that uses computer vision, pose detection, and a work/break timer to help users maintain healthy posture and productivity. It includes:

✅ 1. A Web-Based PosturePal Interface (Flask + HTML)
✅ 2. A Standalone Desktop App (OpenCV + MediaPipe)

Both systems are included in this repository.

📂 Project Structure
PosturePal/
│
├── chatbot.py                 # Flask server for website + chatbot (port 5000)
├── posture_server.py          # Flask webcam posture server (port 8000)
├── cameraposture.py           # Standalone desktop posture application
│
├── index.html                 # Main web interface
├── pose_landmarker_full.task  # MediaPipe pose model
├── posture_settings.json      # Generated user settings + calibration
├── posture_log.csv            # Session log file
│
└── README.md                  # Documentation

🌐 1. Web PosturePal (Flask + HTML)

The web interface allows users to:

Interact with the built-in AI chatbot

Open the camera demo for posture detection

View instructions, UI panels, and modals

🌍 Architecture
Component	File	Port	Purpose
Main Web Server	chatbot.py	5000	Serves UI + chatbot
Posture Camera Server	posture_server.py	8000	Runs webcam posture analysis
Frontend	index.html	(Local file / served via Flask)	UI
🔌 Dual-Server Design
index.html
   │
   ├── fetch() → localhost:5000 (chatbot responses)
   └── window.open() → localhost:8000 (camera posture detection)


Both Flask servers must be running simultaneously.

▶️ How to Run the Web Version
Terminal 1 — Start chatbot + website
python chatbot.py


Open:

http://localhost:5000

Terminal 2 — Start webcam posture server
python posture_server.py


Then click “Start Web Demo” in the UI to launch the camera window.

🧠 2. Standalone PosturePal Desktop App

The full desktop application performs real-time posture detection using OpenCV + MediaPipe with a Pomodoro-style work/break timer.

Run it with:

python cameraposture.py

📸 Real-Time Pose Detection

PosturePal uses MediaPipe PoseLandmarker to detect 33 body landmarks including:

Nose, eyes, ears

Shoulders, elbows

Hips, knees

Runs at 25–30 FPS.

📐 Angle Calculations

It computes three major angles:

Neck angle (forward head posture)

Shoulder angle (rounded shoulders)

Back angle (slouching/spine curvature)

Formula:

angle = arccos((ba · bc) / (|ba| × |bc|))

Smoothing

8-frame median filter removes jitter.

Issue Detection

Angles below thresholds trigger issue states.

🔔 Smart Notification System

After 10 seconds of bad posture:

Windows toast notification

Loud audio alert

Console log entry

Issue state resets when posture improves

⏱️ Work/Break Timer (Pomodoro Style)

Default: 30s work / 5s break

Timer widget

Full-screen break overlay

Pause/resume

Session counter

🎯 Calibration System

Press C to calibrate:

Sit straight → 2 seconds

Captures ~60 frames

Computes your baseline angles

Adjusts thresholds

Saves to posture_settings.json

Skip with S.

🖼️ User Interface Features

Skeleton overlay (green = good, red = bad)

Angle readouts (raw + smoothed)

Posture status panel

Timer widget

Break screen

Help overlay

⌨️ Keyboard Shortcuts
Key	Action
C	Calibrate
S	Skip calibration
SPACE	Pause/resume timer
Q	Quit
H	Toggle help overlay
⚙️ posture_settings.json
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

🔧 Troubleshooting
🟥 Camera Not Opening

Run both Flask servers in separate terminals

Check webcam permissions

Make sure no other app is using the camera

🟥 Port 5000 or 8000 Not Working

You may have stopped one server

Restart both in separate terminals

🟥 Model Not Found

Place:

pose_landmarker_full.task


in the project folder.

🟥 No Notifications

Install win10toast

Maintain bad posture for 10+ seconds

Check Windows notification settings

🧱 System Architecture Overview
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

🚀 Future Improvements

Web-based posture detection (WASM + TensorFlow.js)

Daily/weekly analytics dashboard

AI posture advice

Slack/email summaries

Multi-user support

Animated stretching routines

🔐 Privacy

✔ 100% local
✔ No cloud processing
✔ No uploads
✔ Webcam frames never stored

All posture processing happens directly on your machine.

📄 License & Attribution

Uses open-source technologies:

Google MediaPipe

OpenCV

NumPy

Flask

win10toast
