from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
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

app = Flask(__name__)
CORS(app)

# -------------------------------
# SETTINGS
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

# -------------------------------
# POSTURE ANALYZER
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
        self.calibration_target = 30  # Reduced to 1 second at 30fps
        self.calibrating_samples = []
    
    def calibrate(self, neck_angle, shoulder_angle, back_angle):
        """Calibration mode - simplified"""
        if self.calibration_data["calibration_complete"]:
            return False
        
        # More lenient thresholds for calibration
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
            
            print(f"Calibrating: {self.calibration_frames}/{self.calibration_target} - Neck: {neck_angle:.1f}, Shoulder: {shoulder_angle:.1f}, Back: {back_angle:.1f}")
            
            if self.calibration_frames >= self.calibration_target:
                # Calculate baselines from samples
                neck_samples = [s['neck'] for s in self.calibrating_samples]
                shoulder_samples = [s['shoulder'] for s in self.calibrating_samples]
                back_samples = [s['back'] for s in self.calibrating_samples]
                
                self.calibration_data["neck_baseline"] = float(np.mean(neck_samples))
                self.calibration_data["shoulder_baseline"] = float(np.mean(shoulder_samples))
                self.calibration_data["back_baseline"] = float(np.mean(back_samples))
                self.calibration_data["calibration_complete"] = True
                
                print(f"\n✓ CALIBRATION COMPLETE!")
                print(f"  Neck baseline: {self.calibration_data['neck_baseline']:.1f}°")
                print(f"  Shoulder baseline: {self.calibration_data['shoulder_baseline']:.1f}°")
                print(f"  Back baseline: {self.calibration_data['back_baseline']:.1f}°\n")
                
                return True
        else:
            # Reset if posture is not good
            if self.calibration_frames > 0:
                print(f"Calibration reset - posture not maintained (Neck: {neck_angle:.1f}, Shoulder: {shoulder_angle:.1f}, Back: {back_angle:.1f})")
            self.calibration_frames = 0
            self.calibrating_samples = []
        
        return False
    
    def get_calibration_progress(self):
        if self.calibration_data["calibration_complete"]:
            return 100
        return int((self.calibration_frames / self.calibration_target) * 100)
    
    def reset_calibration(self):
        print("Resetting calibration...")
        self.calibration_frames = 0
        self.calibrating_samples = []
        self.calibration_data = {
            "neck_baseline": None,
            "shoulder_baseline": None,
            "back_baseline": None,
            "calibration_complete": False
        }
    
    def analyze_posture(self, neck_angle, shoulder_angle, back_angle, face_distance, current_time):
        """Analyze posture and return detailed info"""
        
        self.neck_angle_history.append(neck_angle)
        self.shoulder_angle_history.append(shoulder_angle)
        self.back_angle_history.append(back_angle)
        
        smooth_neck = float(np.median(self.neck_angle_history)) if len(self.neck_angle_history) > 0 else neck_angle
        smooth_shoulder = float(np.median(self.shoulder_angle_history)) if len(self.shoulder_angle_history) > 0 else shoulder_angle
        smooth_back = float(np.median(self.back_angle_history)) if len(self.back_angle_history) > 0 else back_angle
        
        # Use calibrated baselines if available, otherwise use defaults
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
        
        # Check neck (forward head)
        is_forward_head = smooth_neck < neck_threshold
        issues["forward_head"] = is_forward_head
        
        if is_forward_head:
            if self.issue_start_times["forward_head"] is None:
                self.issue_start_times["forward_head"] = current_time
                print(f"⚠ Forward head detected (angle: {smooth_neck:.1f}° < {neck_threshold:.1f}°)")
            primary_issue = "forward_head"
            
            duration = current_time - self.issue_start_times["forward_head"]
            if duration > warning_duration and not self.notification_sent["forward_head"]:
                notification_triggered = True
                self.notification_sent["forward_head"] = True
                print(f"🔔 ALERT: Forward head for {duration:.1f}s")
        else:
            self.issue_start_times["forward_head"] = None
            self.notification_sent["forward_head"] = False
        
        # Check shoulders (rounded shoulders)
        is_rounded_shoulders = smooth_shoulder < shoulder_threshold
        issues["rounded_shoulders"] = is_rounded_shoulders
        
        if is_rounded_shoulders:
            if self.issue_start_times["rounded_shoulders"] is None:
                self.issue_start_times["rounded_shoulders"] = current_time
                print(f"⚠ Rounded shoulders detected (angle: {smooth_shoulder:.1f}° < {shoulder_threshold:.1f}°)")
            if primary_issue is None:
                primary_issue = "rounded_shoulders"
            
            duration = current_time - self.issue_start_times["rounded_shoulders"]
            if duration > warning_duration and not self.notification_sent["rounded_shoulders"]:
                notification_triggered = True
                self.notification_sent["rounded_shoulders"] = True
                print(f"🔔 ALERT: Rounded shoulders for {duration:.1f}s")
        else:
            self.issue_start_times["rounded_shoulders"] = None
            self.notification_sent["rounded_shoulders"] = False
        
        # Check back (slouched)
        is_slouched = smooth_back < back_threshold
        issues["slouched_back"] = is_slouched
        
        if is_slouched:
            if self.issue_start_times["slouched_back"] is None:
                self.issue_start_times["slouched_back"] = current_time
                print(f"⚠ Slouched back detected (angle: {smooth_back:.1f}° < {back_threshold:.1f}°)")
            if primary_issue is None:
                primary_issue = "slouched_back"
            
            duration = current_time - self.issue_start_times["slouched_back"]
            if duration > warning_duration and not self.notification_sent["slouched_back"]:
                notification_triggered = True
                self.notification_sent["slouched_back"] = True
                print(f"🔔 ALERT: Slouched back for {duration:.1f}s")
        else:
            self.issue_start_times["slouched_back"] = None
            self.notification_sent["slouched_back"] = False
        
        # Check distance
        too_close = face_distance > settings["face_distance_threshold"]
        issues["too_close"] = too_close
        
        if too_close:
            if self.issue_start_times["too_close"] is None:
                self.issue_start_times["too_close"] = current_time
            
            duration = current_time - self.issue_start_times["too_close"]
            if duration > warning_duration and not self.notification_sent["too_close"]:
                notification_triggered = True
                self.notification_sent["too_close"] = True
                print(f"🔔 ALERT: Too close to screen for {duration:.1f}s")
        else:
            self.issue_start_times["too_close"] = None
            self.notification_sent["too_close"] = False
        
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
    """Calculate angle ABC"""
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
    """Distance between eyes"""
    left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
    right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
    
    eye_distance = np.sqrt((left_eye.x - right_eye.x)**2 + (left_eye.y - right_eye.y)**2)
    return eye_distance

# -------------------------------
# GLOBAL VARIABLES
# -------------------------------
camera = None
posture_analyzer = PostureAnalyzer()
landmarker = None
mp_pose = None
is_calibrating = False

def initialize_mediapipe():
    """Initialize MediaPipe PoseLandmarker"""
    global landmarker, mp_pose
    
    model_path = settings["model_path"]
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found: {model_path}")
        print("Download from: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task")
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
# VIDEO STREAM
# -------------------------------
def generate_frames():
    """Generate video frames with posture analysis"""
    global camera, posture_analyzer, landmarker, mp_pose, is_calibrating
    
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Use time-based timestamp in milliseconds
    frame_count = 0
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Generate monotonically increasing timestamp in milliseconds
        timestamp = int(time.time() * 1000)
        frame_count += 1
        
        # Process frame with MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        try:
            result = landmarker.detect_for_video(mp_image, timestamp)
            
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]
                
                # Get landmark points
                ear_l = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
                shoulder_l = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                hip_l = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                knee_l = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                elbow_l = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                
                # Calculate angles
                neck_angle = calculate_angle(ear_l, shoulder_l, hip_l)
                shoulder_angle = calculate_angle(ear_l, shoulder_l, elbow_l)
                back_angle = calculate_angle(shoulder_l, hip_l, knee_l)
                face_distance = calculate_face_distance(landmarks, mp_pose)
                
                # Handle calibration
                if is_calibrating:
                    posture_analyzer.calibrate(neck_angle, shoulder_angle, back_angle)
                
                # Analyze posture (always run, even during calibration)
                current_time = time.time()
                analysis = posture_analyzer.analyze_posture(
                    neck_angle, shoulder_angle, back_angle, 
                    face_distance, current_time
                )
                
                # Choose color based on posture
                skeleton_color = (0, 0, 255) if analysis["is_bad"] else (0, 255, 0)  # BGR: Red or Green
                
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
                
                # Draw joints
                key_points = [
                    mp_pose.PoseLandmark.LEFT_EAR.value,
                    mp_pose.PoseLandmark.RIGHT_EAR.value,
                    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                    mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                    mp_pose.PoseLandmark.LEFT_HIP.value,
                    mp_pose.PoseLandmark.RIGHT_HIP.value,
                ]
                
                for point in key_points:
                    lm = landmarks[point]
                    pos = (int(lm.x * w), int(lm.y * h))
                    cv2.circle(frame, pos, 8, skeleton_color, -1)
                    cv2.circle(frame, pos, 8, (255, 255, 255), 2)
                
                # Draw status and angles
                status_color = (0, 0, 255) if analysis["is_bad"] else (0, 255, 0)
                status_text = "BAD POSTURE" if analysis["is_bad"] else "GOOD POSTURE"
                
                cv2.putText(frame, status_text, (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3, cv2.LINE_AA)
                
                # Display angles with color coding
                neck_color = (0, 0, 255) if analysis['smooth_angles']['neck'] < analysis['thresholds']['neck'] else (0, 255, 0)
                shoulder_color = (0, 0, 255) if analysis['smooth_angles']['shoulder'] < analysis['thresholds']['shoulder'] else (0, 255, 0)
                back_color = (0, 0, 255) if analysis['smooth_angles']['back'] < analysis['thresholds']['back'] else (0, 255, 0)
                
                cv2.putText(frame, f"Neck: {int(analysis['smooth_angles']['neck'])}deg (>{int(analysis['thresholds']['neck'])})", 
                           (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, neck_color, 2, cv2.LINE_AA)
                cv2.putText(frame, f"Shoulder: {int(analysis['smooth_angles']['shoulder'])}deg (>{int(analysis['thresholds']['shoulder'])})", 
                           (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, shoulder_color, 2, cv2.LINE_AA)
                cv2.putText(frame, f"Back: {int(analysis['smooth_angles']['back'])}deg (>{int(analysis['thresholds']['back'])})", 
                           (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, back_color, 2, cv2.LINE_AA)
                
                # Show calibration status
                if is_calibrating:
                    progress = posture_analyzer.get_calibration_progress()
                    cv2.putText(frame, f"CALIBRATING: {progress}%", (20, h - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, "Hold good posture!", (20, h - 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                elif not posture_analyzer.calibration_data["calibration_complete"]:
                    cv2.putText(frame, "Click CALIBRATE to start", (20, h - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        
        except Exception as e:
            print(f"Error processing frame: {e}")
            cv2.putText(frame, f"Error: {str(e)}", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# -------------------------------
# ROUTES
# -------------------------------
@app.route('/')
def index():
    """Serve the web interface"""
    return render_template('camera_demo.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def get_status():
    """Get current posture status"""
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
    """Start calibration"""
    global is_calibrating
    posture_analyzer.reset_calibration()
    is_calibrating = True
    print("\n🎯 Starting calibration... Please sit with good posture!")
    return jsonify({"status": "calibration_started"})

@app.route('/api/stop_calibrate', methods=['POST'])
def stop_calibration():
    """Stop calibration"""
    global is_calibrating
    is_calibrating = False
    print("⏹ Calibration stopped")
    return jsonify({"status": "calibration_stopped"})

@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings():
    """Get or update settings"""
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
    print("PosturePal Flask Server")
    print("=" * 60)
    print("Initializing MediaPipe...")
    
    if not initialize_mediapipe():
        print("Failed to initialize MediaPipe. Exiting.")
        print("Make sure 'pose_landmarker_full.task' is in the same folder!")
        exit(1)
    
    print("✓ MediaPipe initialized successfully!")
    print("\nServer starting on http://localhost:5001")
    print("Open your browser and navigate to: http://localhost:5001")
    print("\nTips:")
    print("  - Sit with good posture when calibrating")
    print("  - Calibration takes ~1 second of good posture")
    print("  - Watch the console for real-time feedback")
    print("=" * 60)
    print()
    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True, use_reloader=False)
