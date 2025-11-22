import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from datetime import datetime, timedelta
import json
import os
from plyer import notification  # pip install plyer

# -------------------------------
# CONFIG & SETTINGS
# -------------------------------
SETTINGS_FILE = "posture_settings.json"
LOG_FILE = "posture_log.csv"

DEFAULT_SETTINGS = {
    "work_interval": 20 * 60,  # 20 minutes
    "break_duration": 20,  # 20 seconds
    "neck_angle_threshold": 170,
    "shoulder_slouch_threshold": 160,
    "posture_warning_duration": 3,
    "face_distance_threshold": 0.25,  # Normalized distance (closer = larger face)
    "model_path": "pose_landmarker_full.task"
}

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            settings = json.load(f)
            # Merge with defaults for any missing keys
            return {**DEFAULT_SETTINGS, **settings}
    return DEFAULT_SETTINGS.copy()

def save_settings(settings):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=2)

# -------------------------------
# DATA LOGGING
# -------------------------------
class PostureLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.current_session = {
            "start_time": datetime.now().isoformat(),
            "bad_posture_events": [],
            "distance_warnings": 0,
            "breaks_taken": 0
        }
        
        # Create log file with headers if it doesn't exist
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                f.write("timestamp,event_type,duration_seconds,details\n")
    
    def log_event(self, event_type, duration=0, details=""):
        timestamp = datetime.now().isoformat()
        with open(self.log_file, 'a') as f:
            f.write(f"{timestamp},{event_type},{duration},{details}\n")
    
    def log_bad_posture(self, duration, posture_type):
        self.log_event("bad_posture", duration, posture_type)
    
    def log_distance_warning(self):
        self.current_session["distance_warnings"] += 1
        self.log_event("distance_warning", 0, "too_close")
    
    def log_break_taken(self):
        self.current_session["breaks_taken"] += 1
        self.log_event("break_taken", 0, "20-20-20")

# -------------------------------
# NOTIFICATION SYSTEM
# -------------------------------
class NotificationManager:
    def __init__(self):
        self.last_notification_time = 0
        self.notification_cooldown = 10  # Seconds between notifications
    
    def send_notification(self, title, message):
        current_time = time.time()
        if current_time - self.last_notification_time >= self.notification_cooldown:
            try:
                notification.notify(
                    title=title,
                    message=message,
                    app_name="Desk Health Monitor",
                    timeout=5
                )
                self.last_notification_time = current_time
            except Exception as e:
                print(f"Notification error: {e}")

# -------------------------------
# SETTINGS UI
# -------------------------------
class SettingsUI:
    def __init__(self, settings):
        self.settings = settings.copy()
        self.show = False
        self.active_field = None
        self.fields = [
            {"key": "work_interval", "name": "Work Interval (min)", "type": "int", "divisor": 60, "min": 1, "max": 60},
            {"key": "break_duration", "name": "Break Duration (sec)", "type": "int", "divisor": 1, "min": 10, "max": 120},
            {"key": "neck_angle_threshold", "name": "Neck Angle Threshold", "type": "int", "divisor": 1, "min": 140, "max": 180},
            {"key": "shoulder_slouch_threshold", "name": "Shoulder Threshold", "type": "int", "divisor": 1, "min": 140, "max": 180},
            {"key": "posture_warning_duration", "name": "Warning Delay (sec)", "type": "int", "divisor": 1, "min": 1, "max": 10},
            {"key": "face_distance_threshold", "name": "Distance Threshold", "type": "float", "divisor": 1, "min": 0.15, "max": 0.40}
        ]
        self.scroll_offset = 0
    
    def toggle(self):
        self.show = not self.show
    
    def draw(self, frame):
        if not self.show:
            return
        
        h, w, _ = frame.shape
        
        # Semi-transparent overlay
        overlay = frame.copy()
        panel_width = 500
        panel_height = 450
        panel_x = (w - panel_width) // 2
        panel_y = (h - panel_height) // 2
        
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.95, frame, 0.05, 0, frame)
        
        # Border
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (200, 200, 200), 2)
        
        # Title
        cv2.putText(frame, "SETTINGS", (panel_x + 20, panel_y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Fields
        y_offset = panel_y + 80
        for i, field in enumerate(self.fields):
            value = self.settings[field["key"]] / field["divisor"]
            
            # Field name
            cv2.putText(frame, field["name"], (panel_x + 20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Value with +/- buttons
            value_str = f"{value:.2f}" if field["type"] == "float" else f"{int(value)}"
            
            # Draw - button
            button_y = y_offset - 15
            cv2.rectangle(frame, (panel_x + 350, button_y), 
                         (panel_x + 380, button_y + 25), (100, 100, 100), -1)
            cv2.putText(frame, "-", (panel_x + 360, button_y + 18), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Draw value
            cv2.putText(frame, value_str, (panel_x + 390, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw + button
            cv2.rectangle(frame, (panel_x + 450, button_y), 
                         (panel_x + 480, button_y + 25), (100, 100, 100), -1)
            cv2.putText(frame, "+", (panel_x + 457, button_y + 18), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            y_offset += 50
        
        # Buttons
        button_y = panel_y + panel_height - 60
        
        # Save button
        cv2.rectangle(frame, (panel_x + 100, button_y), 
                     (panel_x + 200, button_y + 40), (0, 200, 0), -1)
        cv2.putText(frame, "SAVE", (panel_x + 130, button_y + 27), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Cancel button
        cv2.rectangle(frame, (panel_x + 300, button_y), 
                     (panel_x + 400, button_y + 40), (0, 0, 200), -1)
        cv2.putText(frame, "CANCEL", (panel_x + 315, button_y + 27), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.putText(frame, "Press 'S' to close", (panel_x + 20, panel_y + panel_height - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    def adjust_value(self, field_index, increment):
        field = self.fields[field_index]
        current_value = self.settings[field["key"]]
        
        if field["type"] == "int":
            step = field["divisor"]
            new_value = current_value + (step if increment else -step)
        else:
            step = 0.01
            new_value = current_value + (step if increment else -step)
        
        # Clamp to min/max
        min_val = field["min"] * field["divisor"]
        max_val = field["max"] * field["divisor"]
        self.settings[field["key"]] = max(min_val, min(max_val, new_value))

# -------------------------------
# COMPACT OVERLAY
# -------------------------------
class CompactOverlay:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 250
        self.height = 110
        self.dragging = False
        self.drag_offset_x = 0
        self.drag_offset_y = 0
    
    def draw(self, frame, posture_tracker, timer):
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (self.x, self.y), 
                     (self.x + self.width, self.y + self.height), 
                     (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Border
        cv2.rectangle(frame, (self.x, self.y), 
                     (self.x + self.width, self.y + self.height), 
                     (100, 100, 100), 1)
        
        # Bad posture time
        bad_minutes = posture_tracker.get_total_bad_posture_minutes()
        cv2.putText(frame, f"Bad Posture: {bad_minutes}m", 
                   (self.x + 10, self.y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
        
        # Next break timer
        time_remaining = timer.get_time_remaining()
        time_str = timer.format_time(time_remaining)
        status = "Break" if timer.on_break else "Next Break"
        cv2.putText(frame, f"{status}: {time_str}", 
                   (self.x + 10, self.y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        
        # Play/Pause button
        button_x = self.x + 10
        button_y = self.y + 60
        button_size = 30
        
        if timer.paused:
            # Play button (triangle)
            pts = np.array([[button_x + 5, button_y], 
                           [button_x + 5, button_y + button_size],
                           [button_x + button_size, button_y + button_size//2]], np.int32)
            cv2.fillPoly(frame, [pts], (100, 200, 100))
        else:
            # Pause button (two rectangles)
            cv2.rectangle(frame, (button_x + 5, button_y), 
                         (button_x + 12, button_y + button_size), (200, 200, 100), -1)
            cv2.rectangle(frame, (button_x + 18, button_y), 
                         (button_x + 25, button_y + button_size), (200, 200, 100), -1)
        
        # Settings button (gear icon approximation)
        settings_x = self.x + 50
        cv2.circle(frame, (settings_x, button_y + button_size//2), 12, (150, 150, 150), -1)
        cv2.circle(frame, (settings_x, button_y + button_size//2), 6, (30, 30, 30), -1)
        cv2.putText(frame, "S", (settings_x - 5, button_y + button_size//2 + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def is_inside(self, x, y):
        return (self.x <= x <= self.x + self.width and 
                self.y <= y <= self.y + self.height)
    
    def is_play_pause_clicked(self, x, y):
        button_x = self.x + 10
        button_y = self.y + 60
        return (button_x <= x <= button_x + 30 and 
                button_y <= y <= button_y + 30)

# -------------------------------
# Timer Class
# -------------------------------
class WorkBreakTimer:
    def __init__(self, work_duration, break_duration):
        self.work_duration = work_duration
        self.break_duration = break_duration
        self.work_start_time = time.time()
        self.break_start_time = None
        self.on_break = False
        self.break_acknowledged = False
        self.paused = False
        self.pause_start_time = None
        self.total_paused_time = 0
        
    def toggle_pause(self):
        if self.paused:
            # Resume
            if self.pause_start_time:
                self.total_paused_time += time.time() - self.pause_start_time
            self.paused = False
        else:
            # Pause
            self.pause_start_time = time.time()
            self.paused = True
    
    def check_timer(self):
        if self.paused:
            return "PAUSED"
        
        current_time = time.time()
        
        if not self.on_break:
            elapsed = (current_time - self.work_start_time) - self.total_paused_time
            if elapsed >= self.work_duration:
                self.on_break = True
                self.break_start_time = current_time
                self.break_acknowledged = False
                self.total_paused_time = 0
                return "BREAK_TIME"
            return "WORKING"
        else:
            elapsed = current_time - self.break_start_time
            if elapsed >= self.break_duration and self.break_acknowledged:
                self.on_break = False
                self.work_start_time = current_time
                self.total_paused_time = 0
                return "BREAK_COMPLETE"
            return "ON_BREAK"
    
    def acknowledge_break(self):
        self.break_acknowledged = True
    
    def get_time_remaining(self):
        if self.paused:
            return self.last_time_remaining if hasattr(self, 'last_time_remaining') else 0
        
        current_time = time.time()
        if not self.on_break:
            elapsed = (current_time - self.work_start_time) - self.total_paused_time
            remaining = max(0, self.work_duration - elapsed)
        else:
            elapsed = current_time - self.break_start_time
            remaining = max(0, self.break_duration - elapsed)
        
        self.last_time_remaining = int(remaining)
        return self.last_time_remaining
    
    def format_time(self, seconds):
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"
    
    def update_settings(self, work_duration, break_duration):
        self.work_duration = work_duration
        self.break_duration = break_duration

# -------------------------------
# Posture Tracker
# -------------------------------
class PostureTracker:
    def __init__(self, warning_duration):
        self.bad_posture_start = None
        self.warning_duration = warning_duration
        self.total_bad_posture_time = 0
        self.last_check_time = time.time()
        self.current_bad_posture_type = None
        
    def update(self, is_bad_posture, posture_type=None):
        current_time = time.time()
        
        if is_bad_posture:
            if self.bad_posture_start is None:
                self.bad_posture_start = current_time
                self.current_bad_posture_type = posture_type
            else:
                elapsed = current_time - self.last_check_time
                self.total_bad_posture_time += elapsed
        else:
            self.bad_posture_start = None
            self.current_bad_posture_type = None
        
        self.last_check_time = current_time
        
    def should_show_warning(self):
        if self.bad_posture_start is None:
            return False
        return (time.time() - self.bad_posture_start) >= self.warning_duration
    
    def get_total_bad_posture_minutes(self):
        return int(self.total_bad_posture_time / 60)
    
    def get_bad_posture_duration(self):
        if self.bad_posture_start is None:
            return 0
        return time.time() - self.bad_posture_start

# -------------------------------
# Angle Calculation
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

# -------------------------------
# Distance Calculation
# -------------------------------
def calculate_face_distance(landmarks, mp_pose):
    # Use distance between eyes as proxy for face size/distance
    left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
    right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
    
    eye_distance = np.sqrt((left_eye.x - right_eye.x)**2 + (left_eye.y - right_eye.y)**2)
    return eye_distance

# -------------------------------
# Break Reminder UI
# -------------------------------
def draw_break_reminder(frame):
    h, w, _ = frame.shape
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    messages = [
        "TIME FOR A BREAK!",
        "",
        "20-20-20 Rule:",
        "Look at something 20 feet away",
        "for 20 seconds",
        "",
        "Press SPACE to acknowledge"
    ]
    
    start_y = h // 2 - len(messages) * 20
    for i, msg in enumerate(messages):
        if i == 0:
            color = (0, 255, 255)
            size = 1.5
        elif i in [2]:
            color = (255, 200, 0)
            size = 1.0
        else:
            color = (255, 255, 255)
            size = 0.8
        
        text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, size, 2)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(frame, msg, (text_x, start_y + i * 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, size, color, 2)

# -------------------------------
# Mouse Callback
# -------------------------------
def mouse_callback(event, x, y, flags, param):
    overlay, settings_ui, timer = param
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if overlay.is_play_pause_clicked(x, y):
            timer.toggle_pause()
        elif overlay.is_inside(x, y):
            overlay.dragging = True
            overlay.drag_offset_x = x - overlay.x
            overlay.drag_offset_y = y - overlay.y
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if overlay.dragging:
            overlay.x = x - overlay.drag_offset_x
            overlay.y = y - overlay.drag_offset_y
    
    elif event == cv2.EVENT_LBUTTONUP:
        overlay.dragging = False

# -------------------------------
# MAIN
# -------------------------------
settings = load_settings()
model_path = settings["model_path"]

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    output_segmentation_masks=False
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not found.")
    exit()

# Initialize components
timer = WorkBreakTimer(settings["work_interval"], settings["break_duration"])
posture_tracker = PostureTracker(settings["posture_warning_duration"])
logger = PostureLogger(LOG_FILE)
notif_manager = NotificationManager()
settings_ui = SettingsUI(settings)
compact_overlay = CompactOverlay(10, 10)

# Window setup
cv2.namedWindow("Desk Health Monitor")
cv2.setMouseCallback("Desk Health Monitor", mouse_callback, (compact_overlay, settings_ui, timer))

# Track window focus
window_focused = True
last_focus_check = time.time()
last_distance_warning_time = 0

with PoseLandmarker.create_from_options(options) as landmarker:
    timestamp = 0
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        result = landmarker.detect_for_video(mp_image, timestamp)
        timestamp += 33

        h, w, _ = frame.shape
        is_bad_posture = False
        posture_type = None

        # Check window focus
        current_time = time.time()
        if current_time - last_focus_check > 1.0:
            try:
                window_name = cv2.getWindowProperty("Desk Health Monitor", cv2.WND_PROP_VISIBLE)
                window_focused = window_name >= 1
            except:
                window_focused = False
            last_focus_check = current_time

        # Posture analysis
        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            
            try:
                # Neck angle
                ear_l = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
                shoulder_l = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                hip_l = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                neck_angle = calculate_angle(ear_l, shoulder_l, hip_l)
                
                # Shoulder angle
                elbow_l = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                shoulder_angle = calculate_angle(ear_l, shoulder_l, elbow_l)
                
                # Distance check
                face_distance = calculate_face_distance(landmarks, mp_pose)
                too_close = face_distance > settings["face_distance_threshold"]
                
                if too_close and current_time - last_distance_warning_time > 5:
                    cv2.putText(frame, "TOO CLOSE TO SCREEN!", 
                               (w // 2 - 200, h - 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
                    logger.log_distance_warning()
                    if not window_focused:
                        notif_manager.send_notification("Distance Warning", 
                                                       "You're too close to the screen!")
                    last_distance_warning_time = current_time
                
                # Posture checks
                if neck_angle < settings["neck_angle_threshold"]:
                    is_bad_posture = True
                    posture_type = "forward_head"
                    if posture_tracker.should_show_warning():
                        cv2.putText(frame, "HEAD FORWARD! Sit Up Straight!", 
                                   (w // 2 - 250, 120), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                        if not window_focused:
                            notif_manager.send_notification("Posture Alert", 
                                                           "Head forward - sit up straight!")
                
                if shoulder_angle < settings["shoulder_slouch_threshold"]:
                    is_bad_posture = True
                    posture_type = "rounded_shoulders"
                    if posture_tracker.should_show_warning():
                        cv2.putText(frame, "SHOULDERS ROUNDED! Roll them back!", 
                                   (w // 2 - 280, 160), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 100, 255), 3)
                        if not window_focused:
                            notif_manager.send_notification("Posture Alert", 
                                                           "Rounded shoulders detected!")
                
                # Display angles
                shoulder_x = int(shoulder_l.x * w)
                shoulder_y = int(shoulder_l.y * h)
                
                neck_color = (0, 255, 0) if neck_angle >= settings["neck_angle_threshold"] else (0, 0, 255)
                cv2.putText(frame, f'Neck: {int(neck_angle)}', 
                           (shoulder_x + 10, shoulder_y - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, neck_color, 2)
                
                shoulder_color = (0, 255, 0) if shoulder_angle >= settings["shoulder_slouch_threshold"] else (0, 0, 255)
                cv2.putText(frame, f'Shoulder: {int(shoulder_angle)}', 
                           (shoulder_x + 10, shoulder_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, shoulder_color, 2)
                
                # Draw pose landmarks
                mp_drawing.draw_landmarks(
                    frame, result.pose_landmarks[0], POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )

            except Exception as e:
                pass
        
        # Update posture tracker
        posture_tracker.update(is_bad_posture, posture_type)
        
        # Log bad posture events
        if is_bad_posture and posture_tracker.should_show_warning():
            duration = posture_tracker.get_bad_posture_duration()
            if duration > 5:  # Log if bad posture lasts more than 5 seconds
                logger.log_bad_posture(duration, posture_type or "unknown")
        
        # Check break timer
        timer_status = timer.check_timer()
        
        if timer_status in ["BREAK_TIME", "ON_BREAK"]:
            if not timer.break_acknowledged:
                draw_break_reminder(frame)
                if not window_focused and timer_status == "BREAK_TIME":
                    notif_manager.send_notification("Break Time!", 
                                                   "Time for your 20-20-20 break!")
        
        # Draw compact overlay
        compact_overlay.draw(frame, posture_tracker, timer)
        
        # Draw settings UI
        settings_ui.draw(frame)
        
        cv2.imshow("Desk Health Monitor", frame)
        
        # Keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            if timer.on_break:
                timer.acknowledge_break()
                logger.log_break_taken()
        elif key == ord('s') or key == ord('S'):
            if settings_ui.show:
                # Save settings
                save_settings(settings_ui.settings)
                settings = settings_ui.settings.copy()
                timer.update_settings(settings["work_interval"], settings["break_duration"])
                posture_tracker.warning_duration = settings["posture_warning_duration"]
            settings_ui.toggle()
        elif key == ord('+') or key == ord('='):
            if settings_ui.show:
                for i in range(len(settings_ui.fields)):
                    settings_ui.adjust_value(i, True)
        elif key == ord('-') or key == ord('_'):
            if settings_ui.show:
                for i in range(len(settings_ui.fields)):
                    settings_ui.adjust_value(i, False)
        elif key >= ord('1') and key <= ord('6'):
            if settings_ui.show:
                field_index = key - ord('1')
                if field_index < len(settings_ui.fields):
                    settings_ui.adjust_value(field_index, True)

cap.release()
cv2.destroyAllWindows()
logger.log_event("session_end", int(time.time() - posture_tracker.last_check_time), "normal")