from flask import Flask, request, jsonify, render_template, session, Response
from flask_cors import CORS
from openai import OpenAI
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import json
import os
from collections import deque
import threading

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "abc123"
CORS(app, supports_credentials=True)

# Initialize OpenAI client
client = OpenAI(api_key="APIKEYHERE")

# -------------------------------
# POSTURE DETECTION GLOBALS
# -------------------------------
SETTINGS_FILE = "posture_settings.json"

DEFAULT_SETTINGS = {
    "work_interval": 30,
    "break_duration": 5,
    "neck_angle_threshold": 160,
    "shoulder_slouch_threshold": 155,
    "back_angle_threshold": 155,
    "posture_warning_duration": 10,
    "face_distance_threshold": 0.25,
    "model_path": "pose_landmarker_full.task",
}

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            settings = json.load(f)
            return {**DEFAULT_SETTINGS, **settings}
    return DEFAULT_SETTINGS.copy()

settings = load_settings()
camera = None
posture_analyzer = None
landmarker = None
mp_pose = None
is_calibrating = False

# -------------------------------
# POSTURE ANALYZER CLASS
# -------------------------------
class PostureAnalyzer:
    def __init__(self):
        self.neck_angle_history = deque(maxlen=8)
        self.shoulder_angle_history = deque(maxlen=8)
        self.back_angle_history = deque(maxlen=8)
        
        self.issue_start_times = {
            "forward_head": None,
            "rounded_shoulders": None,
            "slouched_back": None,
            "too_close": None
        }
        
        self.notification_sent = {
            "forward_head": False,
            "rounded_shoulders": False,
            "slouched_back": False,
            "too_close": False
        }
        
        self.calibration_data = {
            "neck_baseline": None,
            "shoulder_baseline": None,
            "back_baseline": None,
            "calibration_complete": False
        }
        self.calibration_frames = 0
        self.calibration_target = 60
        self.calibrating_samples = []
    
    def calibrate(self, neck_angle, shoulder_angle, back_angle):
        if self.calibration_data["calibration_complete"]:
            return False
        
        good_neck = 160 <= neck_angle <= 180
        good_shoulder = 155 <= shoulder_angle <= 180
        good_back = 155 <= back_angle <= 180
        
        if good_neck and good_shoulder and good_back:
            self.calibration_frames += 1
            self.calibrating_samples.append({
                'neck': neck_angle,
                'shoulder': shoulder_angle,
                'back': back_angle
            })
            
            if self.calibration_frames >= self.calibration_target:
                neck_samples = [s['neck'] for s in self.calibrating_samples]
                shoulder_samples = [s['shoulder'] for s in self.calibrating_samples]
                back_samples = [s['back'] for s in self.calibrating_samples]
                
                self.calibration_data["neck_baseline"] = float(np.mean(neck_samples))
                self.calibration_data["shoulder_baseline"] = float(np.mean(shoulder_samples))
                self.calibration_data["back_baseline"] = float(np.mean(back_samples))
                self.calibration_data["calibration_complete"] = True
                
                print(f"\n✓ CALIBRATION COMPLETE!")
                return True
        else:
            if self.calibration_frames > 0:
                print(f"Calibration reset - posture not maintained")
            self.calibration_frames = 0
            self.calibrating_samples = []
        
        return False
    
    def get_calibration_progress(self):
        if self.calibration_data["calibration_complete"]:
            return 100
        return int((self.calibration_frames / self.calibration_target) * 100)
    
    def reset_calibration(self):
        self.calibration_frames = 0
        self.calibrating_samples = []
        self.calibration_data = {
            "neck_baseline": None,
            "shoulder_baseline": None,
            "back_baseline": None,
            "calibration_complete": False
        }
    
    def analyze_posture(self, neck_angle, shoulder_angle, back_angle, face_distance, current_time):
        self.neck_angle_history.append(neck_angle)
        self.shoulder_angle_history.append(shoulder_angle)
        self.back_angle_history.append(back_angle)
        
        smooth_neck = float(np.median(self.neck_angle_history)) if len(self.neck_angle_history) > 0 else neck_angle
        smooth_shoulder = float(np.median(self.shoulder_angle_history)) if len(self.shoulder_angle_history) > 0 else shoulder_angle
        smooth_back = float(np.median(self.back_angle_history)) if len(self.back_angle_history) > 0 else back_angle
        
        neck_threshold = settings["neck_angle_threshold"]
        shoulder_threshold = settings["shoulder_slouch_threshold"]
        back_threshold = settings["back_angle_threshold"]
        
        if self.calibration_data["calibration_complete"]:
            neck_threshold = self.calibration_data["neck_baseline"] - 10
            shoulder_threshold = self.calibration_data["shoulder_baseline"] - 10
            back_threshold = self.calibration_data["back_baseline"] - 10
        
        issues = {}
        primary_issue = None
        notification_triggered = False
        warning_duration = settings["posture_warning_duration"]
        
        # Check neck
        is_forward_head = smooth_neck < neck_threshold
        issues["forward_head"] = is_forward_head
        
        if is_forward_head:
            if self.issue_start_times["forward_head"] is None:
                self.issue_start_times["forward_head"] = current_time
            primary_issue = "forward_head"
            
            duration = current_time - self.issue_start_times["forward_head"]
            if duration > warning_duration and not self.notification_sent["forward_head"]:
                notification_triggered = True
                self.notification_sent["forward_head"] = True
        else:
            self.issue_start_times["forward_head"] = None
            self.notification_sent["forward_head"] = False
        
        # Similar logic for shoulders and back...
        is_rounded_shoulders = smooth_shoulder < shoulder_threshold
        issues["rounded_shoulders"] = is_rounded_shoulders
        
        is_slouched = smooth_back < back_threshold
        issues["slouched_back"] = is_slouched
        
        too_close = face_distance > settings["face_distance_threshold"]
        issues["too_close"] = too_close
        
        is_bad_posture = any(issues.values())
        
        return {
            "issues": issues,
            "is_bad": is_bad_posture,
            "primary_issue": primary_issue,
            "notification_triggered": notification_triggered,
            "smooth_angles": {
                "neck": smooth_neck,
                "shoulder": smooth_shoulder,
                "back": smooth_back
            },
            "raw_angles": {
                "neck": float(neck_angle),
                "shoulder": float(shoulder_angle),
                "back": float(back_angle)
            },
            "thresholds": {
                "neck": float(neck_threshold),
                "shoulder": float(shoulder_threshold),
                "back": float(back_threshold)
            }
        }

# -------------------------------
# ANGLE CALCULATIONS
# -------------------------------
def calculate_angle(a_coords, b_coords, c_coords):
    a = np.array([a_coords.x, a_coords.y])
    b = np.array([b_coords.x, b_coords.y])
    c = np.array([c_coords.x, c_coords.y])
    
    ba = a - b
    bc = c - b
    
    angle_rad = np.arctan2(bc[1], bc[0]) - np.arctan2(ba[1], ba[0])
    angle_deg = np.abs(np.degrees(angle_rad))
    
    if angle_deg > 180.0:
        angle_deg = 360 - angle_deg
    
    return angle_deg

def calculate_face_distance(landmarks, mp_pose):
    left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
    right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
    
    eye_distance = np.sqrt((left_eye.x - right_eye.x)**2 + (left_eye.y - right_eye.y)**2)
    return eye_distance

# -------------------------------
# INITIALIZE MEDIAPIPE
# -------------------------------
def initialize_mediapipe():
    global landmarker, mp_pose
    
    model_path = settings["model_path"]
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found: {model_path}")
        return False
    
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        output_segmentation_masks=False
    )
    
    landmarker = PoseLandmarker.create_from_options(options)
    mp_pose = mp.solutions.pose
    
    return True

# -------------------------------
# VIDEO GENERATION
# -------------------------------
def generate_frames():
    global camera, posture_analyzer, landmarker, mp_pose, is_calibrating
    
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    timestamp = 0
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        timestamp = int(time.time() * 1000)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        try:
            result = landmarker.detect_for_video(mp_image, timestamp)
            
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]
                
                ear_l = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
                shoulder_l = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                hip_l = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                knee_l = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                elbow_l = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                
                neck_angle = calculate_angle(ear_l, shoulder_l, hip_l)
                shoulder_angle = calculate_angle(ear_l, shoulder_l, elbow_l)
                back_angle = calculate_angle(shoulder_l, hip_l, knee_l)
                face_distance = calculate_face_distance(landmarks, mp_pose)
                
                if is_calibrating:
                    posture_analyzer.calibrate(neck_angle, shoulder_angle, back_angle)
                
                current_time = time.time()
                analysis = posture_analyzer.analyze_posture(
                    neck_angle, shoulder_angle, back_angle, 
                    face_distance, current_time
                )
                
                skeleton_color = (0, 0, 255) if analysis["is_bad"] else (0, 255, 0)
                
                # Draw skeleton
                connections = [
                    (mp_pose.PoseLandmark.LEFT_EAR.value, mp_pose.PoseLandmark.LEFT_SHOULDER.value),
                    (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value),
                    (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value),
                    (mp_pose.PoseLandmark.RIGHT_EAR.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
                    (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_HIP.value),
                    (mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value),
                    (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
                ]
                
                for start, end in connections:
                    start_lm = landmarks[start]
                    end_lm = landmarks[end]
                    start_pos = (int(start_lm.x * w), int(start_lm.y * h))
                    end_pos = (int(end_lm.x * w), int(end_lm.y * h))
                    cv2.line(frame, start_pos, end_pos, skeleton_color, 3)
                
                # Draw status
                status_color = (0, 0, 255) if analysis["is_bad"] else (0, 255, 0)
                status_text = "BAD POSTURE" if analysis["is_bad"] else "GOOD POSTURE"
                
                cv2.putText(frame, status_text, (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3, cv2.LINE_AA)
        
        except Exception as e:
            print(f"Error processing frame: {e}")
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# -------------------------------
# ROUTES - MAIN PAGE
# -------------------------------
@app.route('/')
def index():
    return render_template('newnewindex.html')

# -------------------------------
# ROUTES - CHATBOT
# -------------------------------
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    if "conversation" not in session:
        session["conversation"] = [
            {"role": "system", "content": "You are a helpful, friendly medical assistant, designed to help with posture and exercises that help with everyday health."}
        ]

    session["conversation"].append({"role": "user", "content": user_message})

    instructions = "You are a helpful, friendly medical assistant, designed to help with posture and exercises that help with everyday health, print only in point form, under 40 words."
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=session["conversation"]
    )

    bot_reply = response.choices[0].message.content

    session["conversation"].append({"role": "assistant", "content": bot_reply})

    return jsonify({"reply": bot_reply})

@app.route("/clear", methods=["POST"])
def clear():
    session.pop("conversation", None)
    return jsonify({"status": "Conversation cleared."})

# -------------------------------
# ROUTES - POSTURE CAMERA
# -------------------------------
@app.route('/camera')
def camera_page():
    return render_template('camera_demo.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def get_status():
    return jsonify({
        "calibrated": posture_analyzer.calibration_data["calibration_complete"],
        "calibration_progress": posture_analyzer.get_calibration_progress(),
        "is_calibrating": is_calibrating,
        "baselines": {
            "neck": posture_analyzer.calibration_data.get("neck_baseline"),
            "shoulder": posture_analyzer.calibration_data.get("shoulder_baseline"),
            "back": posture_analyzer.calibration_data.get("back_baseline")
        }
    })

@app.route('/api/calibrate', methods=['POST'])
def start_calibration():
    global is_calibrating
    posture_analyzer.reset_calibration()
    is_calibrating = True
    print("\n🎯 Starting calibration...")
    return jsonify({"status": "calibration_started"})

@app.route('/api/stop_calibrate', methods=['POST'])
def stop_calibration():
    global is_calibrating
    is_calibrating = False
    print("⏹ Calibration stopped")
    return jsonify({"status": "calibration_stopped"})

@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings():
    global settings
    if request.method == 'POST':
        settings = request.json
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
        return jsonify({"status": "settings_updated"})
    else:
        return jsonify(settings)

# -------------------------------
# MAIN
# -------------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("PosturePal Unified Server")
    print("=" * 60)
    print("Initializing MediaPipe...")
    
    if not initialize_mediapipe():
        print("Failed to initialize MediaPipe. Exiting.")
        exit(1)
    
    posture_analyzer = PostureAnalyzer()
    
    print("✓ MediaPipe initialized!")
    print("\nServer starting on http://localhost:5000")
    print("  - Main page: http://localhost:5000")
    print("  - Camera: http://localhost:5000/camera")
    print("  - Chatbot: Integrated in main page")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)
