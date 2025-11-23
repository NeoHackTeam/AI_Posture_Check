import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from datetime import datetime, timedelta
import json
import os
from collections import deque
import winsound  # For Windows; use os.system('afplay /System/Library/Sounds/Glass.aiff') on Mac
from win10toast import ToastNotifier  # Windows notifications

# -------------------------------
# CONFIG & SETTINGS
# -------------------------------
SETTINGS_FILE = "posture_settings.json"
LOG_FILE = "posture_log.csv"

COLORS = {
    "primary_blue": (200, 170, 80),
    "light_blue": (230, 200, 150),
    "accent_blue": (220, 180, 100),
    "white": (255, 255, 255),
    "light_gray": (245, 245, 245),
    "dark_gray": (60, 60, 60),
    "green": (100, 200, 100),
    "red": (100, 100, 255),
    "warning_orange": (100, 180, 255)
}

DEFAULT_SETTINGS = {
    "work_interval": 30,              # 30 seconds for demo
    "break_duration": 5,              # 5 seconds for demo
    "neck_angle_threshold": 155,
    "shoulder_slouch_threshold": 145,
    "back_angle_threshold": 145,
    "posture_warning_duration": 10,   # Notify after 10 seconds
    "face_distance_threshold": 0.25,
    "model_path": "pose_landmarker_full.task",
    "show_landmarks": True,
    "minimal_mode": False,
    "sound_enabled": True,
    "enable_calibration": True,
}

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            settings = json.load(f)
            return {**DEFAULT_SETTINGS, **settings}
    return DEFAULT_SETTINGS.copy()

def save_settings(settings):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=2)

def play_sound():
    """Cross-platform sound alert"""
    try:
        if os.name == 'nt':  # Windows
            winsound.Beep(1000, 500)
        else:  # Mac/Linux
            os.system('afplay /System/Library/Sounds/Glass.aiff')
    except:
        pass

# -------------------------------
# TIMER MANAGER
# -------------------------------
class TimerManager:
    def __init__(self, work_duration, break_duration, sound_enabled=True):
        self.work_duration = work_duration
        self.break_duration = break_duration
        self.sound_enabled = sound_enabled
        
        self.is_working = True
        self.session_start = time.time()
        self.current_phase_start = time.time()
        self.paused = False
        self.pause_time = 0
        
        self.total_work_time = 0
        self.total_break_time = 0
        self.sessions_completed = 0
    
    def get_elapsed_time(self):
        """Get elapsed time in current phase"""
        if self.paused:
            return self.pause_time
        elapsed = time.time() - self.current_phase_start
        return elapsed
    
    def get_remaining_time(self):
        """Get remaining time in current phase"""
        if self.is_working:
            remaining = self.work_duration - self.get_elapsed_time()
        else:
            remaining = self.break_duration - self.get_elapsed_time()
        return max(0, remaining)
    
    def get_phase_status(self):
        """Check if phase should switch"""
        elapsed = self.get_elapsed_time()
        if self.is_working and elapsed >= self.work_duration:
            self.switch_to_break()
            return "switched_to_break"
        elif not self.is_working and elapsed >= self.break_duration:
            self.switch_to_work()
            return "switched_to_work"
        return "ongoing"
    
    def switch_to_break(self):
        """Switch to break phase"""
        if self.is_working:
            self.total_work_time += self.work_duration
            self.is_working = False
            self.current_phase_start = time.time()
            self.paused = False
            if self.sound_enabled:
                play_sound()
    
    def switch_to_work(self):
        """Switch to work phase"""
        if not self.is_working:
            self.total_break_time += self.break_duration
            self.sessions_completed += 1
            self.is_working = True
            self.current_phase_start = time.time()
            self.paused = False
            if self.sound_enabled:
                play_sound()
    
    def toggle_pause(self):
        """Pause/resume timer"""
        if self.paused:
            self.current_phase_start = time.time() - self.pause_time
            self.paused = False
        else:
            self.pause_time = self.get_elapsed_time()
            self.paused = True
    
    def format_time(self, seconds):
        """Format seconds to MM:SS"""
        mins = int(seconds) // 60
        secs = int(seconds) % 60
        return f"{mins:02d}:{secs:02d}"
    
    def get_phase_name(self):
        return "WORK" if self.is_working else "BREAK"

# -------------------------------
# ADVANCED POSTURE ANALYZER
# -------------------------------
class AdvancedPostureAnalyzer:
    def __init__(self, calibration_enabled=True):
        self.calibration_enabled = calibration_enabled
        self.calibration_data = {
            "neck_baseline": None,
            "shoulder_baseline": None,
            "back_baseline": None,
            "calibration_complete": False
        }
        self.calibration_frames = 0
        self.calibration_target = 60
        
        self.neck_angle_history = deque(maxlen=8)
        self.shoulder_angle_history = deque(maxlen=8)
        self.back_angle_history = deque(maxlen=8)
        self.posture_issues = {
            "forward_head": False,
            "rounded_shoulders": False,
            "slouched_back": False,
            "too_close": False
        }
        
        self.posture_durations = {
            "forward_head": 0,
            "rounded_shoulders": 0,
            "slouched_back": 0,
            "too_close": 0
        }
        
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
        
    def calibrate(self, neck_angle, shoulder_angle, back_angle):
        """Calibration mode - user sits with good posture for 2 seconds"""
        if not self.calibration_enabled or self.calibration_data["calibration_complete"]:
            return False
        
        good_neck = 155 <= neck_angle <= 180
        good_shoulder = 145 <= shoulder_angle <= 180
        good_back = 145 <= back_angle <= 180
        
        if good_neck and good_shoulder and good_back:
            self.calibration_frames += 1
            
            if self.calibration_data["neck_baseline"] is None:
                self.calibration_data["neck_baseline"] = []
                self.calibration_data["shoulder_baseline"] = []
                self.calibration_data["back_baseline"] = []
            
            self.calibration_data["neck_baseline"].append(neck_angle)
            self.calibration_data["shoulder_baseline"].append(shoulder_angle)
            self.calibration_data["back_baseline"].append(back_angle)
            
            if self.calibration_frames >= self.calibration_target:
                self.calibration_data["neck_baseline"] = np.mean(self.calibration_data["neck_baseline"])
                self.calibration_data["shoulder_baseline"] = np.mean(self.calibration_data["shoulder_baseline"])
                self.calibration_data["back_baseline"] = np.mean(self.calibration_data["back_baseline"])
                self.calibration_data["calibration_complete"] = True
                return True
        else:
            self.calibration_frames = 0
            self.calibration_data["neck_baseline"] = None
            self.calibration_data["shoulder_baseline"] = None
            self.calibration_data["back_baseline"] = None
        
        return False
    
    def get_calibration_progress(self):
        """Returns 0-100 calibration progress"""
        if self.calibration_data["calibration_complete"]:
            return 100
        return int((self.calibration_frames / self.calibration_target) * 100)
    
    def analyze_posture(self, neck_angle, shoulder_angle, back_angle, face_distance, 
                       settings, current_time):
        """Analyze posture using smoothed angles and return detailed info"""
        
        self.neck_angle_history.append(neck_angle)
        self.shoulder_angle_history.append(shoulder_angle)
        self.back_angle_history.append(back_angle)
        
        smooth_neck = np.median(self.neck_angle_history) if len(self.neck_angle_history) > 0 else neck_angle
        smooth_shoulder = np.median(self.shoulder_angle_history) if len(self.shoulder_angle_history) > 0 else shoulder_angle
        smooth_back = np.median(self.back_angle_history) if len(self.back_angle_history) > 0 else back_angle
        
        neck_threshold = settings["neck_angle_threshold"]
        shoulder_threshold = settings["shoulder_slouch_threshold"]
        back_threshold = settings.get("back_angle_threshold", 160)
        
        if self.calibration_data["calibration_complete"]:
            neck_threshold = self.calibration_data["neck_baseline"] - 5
            shoulder_threshold = self.calibration_data["shoulder_baseline"] - 5
            back_threshold = self.calibration_data["back_baseline"] - 5
        
        issues = {}
        primary_issue = None
        notification_triggered = False
        warning_duration = settings["posture_warning_duration"]
        
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
            if self.issue_start_times["forward_head"] is not None:
                duration = current_time - self.issue_start_times["forward_head"]
                self.posture_durations["forward_head"] += duration
            self.issue_start_times["forward_head"] = None
            self.notification_sent["forward_head"] = False
        
        is_rounded_shoulders = smooth_shoulder < shoulder_threshold
        issues["rounded_shoulders"] = is_rounded_shoulders
        
        if is_rounded_shoulders:
            if self.issue_start_times["rounded_shoulders"] is None:
                self.issue_start_times["rounded_shoulders"] = current_time
            if primary_issue is None:
                primary_issue = "rounded_shoulders"
            
            duration = current_time - self.issue_start_times["rounded_shoulders"]
            if duration > warning_duration and not self.notification_sent["rounded_shoulders"]:
                notification_triggered = True
                self.notification_sent["rounded_shoulders"] = True
        else:
            if self.issue_start_times["rounded_shoulders"] is not None:
                duration = current_time - self.issue_start_times["rounded_shoulders"]
                self.posture_durations["rounded_shoulders"] += duration
            self.issue_start_times["rounded_shoulders"] = None
            self.notification_sent["rounded_shoulders"] = False
        
        is_slouched = smooth_back < back_threshold
        issues["slouched_back"] = is_slouched
        
        if is_slouched:
            if self.issue_start_times["slouched_back"] is None:
                self.issue_start_times["slouched_back"] = current_time
            if primary_issue is None:
                primary_issue = "slouched_back"
            
            duration = current_time - self.issue_start_times["slouched_back"]
            if duration > warning_duration and not self.notification_sent["slouched_back"]:
                notification_triggered = True
                self.notification_sent["slouched_back"] = True
        else:
            if self.issue_start_times["slouched_back"] is not None:
                duration = current_time - self.issue_start_times["slouched_back"]
                self.posture_durations["slouched_back"] += duration
            self.issue_start_times["slouched_back"] = None
            self.notification_sent["slouched_back"] = False
        
        too_close = face_distance > settings["face_distance_threshold"]
        issues["too_close"] = too_close
        
        if too_close:
            if self.issue_start_times["too_close"] is None:
                self.issue_start_times["too_close"] = current_time
            
            duration = current_time - self.issue_start_times["too_close"]
            if duration > warning_duration and not self.notification_sent["too_close"]:
                notification_triggered = True
                self.notification_sent["too_close"] = True
        else:
            if self.issue_start_times["too_close"] is not None:
                duration = current_time - self.issue_start_times["too_close"]
                self.posture_durations["too_close"] += duration
            self.issue_start_times["too_close"] = None
            self.notification_sent["too_close"] = False
        
        is_bad_posture = any(issues.values())
        
        return {
            "issues": issues,
            "is_bad": is_bad_posture,
            "primary_issue": primary_issue,
            "notification_triggered": notification_triggered,
            "notification_issue": primary_issue,
            "smooth_angles": {
                "neck": smooth_neck,
                "shoulder": smooth_shoulder,
                "back": smooth_back
            },
            "raw_angles": {
                "neck": neck_angle,
                "shoulder": shoulder_angle,
                "back": back_angle
            },
            "thresholds": {
                "neck": neck_threshold,
                "shoulder": shoulder_threshold,
                "back": back_threshold
            }
        }
    
    def get_issue_duration(self, issue_type):
        """Get how long current issue has been active"""
        if self.issue_start_times[issue_type] is not None:
            return time.time() - self.issue_start_times[issue_type]
        return 0
    
    def get_total_issue_time(self, issue_type):
        """Get total accumulated time for an issue"""
        return self.posture_durations[issue_type]

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
    """Distance between eyes (normalized)"""
    left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
    right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
    
    eye_distance = np.sqrt((left_eye.x - right_eye.x)**2 + (left_eye.y - right_eye.y)**2)
    return eye_distance

def calculate_back_angle(shoulder, hip, knee):
    """Calculate spine curvature via shoulder-hip-knee angle"""
    return calculate_angle(shoulder, hip, knee)

# -------------------------------
# DRAWING UTILITIES
# -------------------------------
def draw_modern_panel(frame, x, y, width, height, color=COLORS["light_blue"], alpha=0.15, border_width=2):
    """Draw modern panel"""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x, y), (x + width, y + height), COLORS["primary_blue"], border_width)

def put_text(frame, text, org, font_scale=0.6, color=COLORS["dark_gray"], thickness=2):
    """Put text with antialiasing"""
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def draw_timer_widget(frame, timer, w, h):
    """Draw work/break timer widget"""
    panel_x = w - 300
    panel_y = 20
    panel_w = 280
    panel_h = 120
    
    draw_modern_panel(frame, panel_x, panel_y, panel_w, panel_h,
                     color=COLORS["light_blue"], alpha=0.2, border_width=2)
    
    # Phase name
    phase = timer.get_phase_name()
    phase_color = COLORS["green"] if timer.is_working else COLORS["warning_orange"]
    put_text(frame, phase, (panel_x + 80, panel_y + 35),
            font_scale=0.9, color=phase_color, thickness=2)
    
    # Time remaining
    remaining = timer.get_remaining_time()
    time_str = timer.format_time(remaining)
    put_text(frame, time_str, (panel_x + 50, panel_y + 75),
            font_scale=1.2, color=COLORS["primary_blue"], thickness=2)
    
    # Pause indicator
    pause_text = "⏸ PAUSED" if timer.paused else ""
    if pause_text:
        put_text(frame, pause_text, (panel_x + 60, panel_y + 105),
                font_scale=0.5, color=COLORS["red"], thickness=1)
    
    # Sessions completed
    put_text(frame, f"✓ Sessions: {timer.sessions_completed}", (panel_x + 10, h - 30),
            font_scale=0.5, color=COLORS["dark_gray"], thickness=1)

def draw_posture_analysis_panel(frame, analysis, w, h):
    """Draw detailed posture analysis on screen"""
    panel_x = 20
    panel_y = 20
    panel_w = 380
    panel_h = 280
    
    draw_modern_panel(frame, panel_x, panel_y, panel_w, panel_h, 
                     color=COLORS["light_blue"], alpha=0.2, border_width=2)
    
    put_text(frame, "POSTURE ANALYSIS", (panel_x + 15, panel_y + 30), 
            font_scale=0.75, color=COLORS["primary_blue"], thickness=2)
    
    y_offset = panel_y + 60
    line_height = 35
    
    # NECK
    neck_raw = analysis["raw_angles"]["neck"]
    neck_smooth = analysis["smooth_angles"]["neck"]
    neck_threshold = analysis["thresholds"]["neck"]
    neck_status = "✓" if neck_raw >= neck_threshold else "✗"
    neck_color = COLORS["green"] if neck_raw >= neck_threshold else COLORS["red"]
    
    put_text(frame, f"Neck {neck_status}", (panel_x + 15, y_offset), 
            font_scale=0.65, color=COLORS["dark_gray"], thickness=2)
    put_text(frame, f"Raw: {int(neck_raw)}° | Smooth: {int(neck_smooth)}°", 
            (panel_x + 150, y_offset), font_scale=0.55, color=neck_color, thickness=1)
    put_text(frame, f"Threshold: {int(neck_threshold)}°", 
            (panel_x + 15, y_offset + 20), font_scale=0.5, color=COLORS["dark_gray"], thickness=1)
    
    # SHOULDERS
    y_offset += line_height + 10
    shoulder_raw = analysis["raw_angles"]["shoulder"]
    shoulder_smooth = analysis["smooth_angles"]["shoulder"]
    shoulder_threshold = analysis["thresholds"]["shoulder"]
    shoulder_status = "✓" if shoulder_raw >= shoulder_threshold else "✗"
    shoulder_color = COLORS["green"] if shoulder_raw >= shoulder_threshold else COLORS["red"]
    
    put_text(frame, f"Shoulders {shoulder_status}", (panel_x + 15, y_offset), 
            font_scale=0.65, color=COLORS["dark_gray"], thickness=2)
    put_text(frame, f"Raw: {int(shoulder_raw)}° | Smooth: {int(shoulder_smooth)}°", 
            (panel_x + 150, y_offset), font_scale=0.55, color=shoulder_color, thickness=1)
    put_text(frame, f"Threshold: {int(shoulder_threshold)}°", 
            (panel_x + 15, y_offset + 20), font_scale=0.5, color=COLORS["dark_gray"], thickness=1)
    
    # BACK
    y_offset += line_height + 10
    back_raw = analysis["raw_angles"]["back"]
    back_smooth = analysis["smooth_angles"]["back"]
    back_threshold = analysis["thresholds"]["back"]
    back_status = "✓" if back_raw >= back_threshold else "✗"
    back_color = COLORS["green"] if back_raw >= back_threshold else COLORS["red"]
    
    put_text(frame, f"Back {back_status}", (panel_x + 15, y_offset), 
            font_scale=0.65, color=COLORS["dark_gray"], thickness=2)
    put_text(frame, f"Raw: {int(back_raw)}° | Smooth: {int(back_smooth)}°", 
            (panel_x + 150, y_offset), font_scale=0.55, color=back_color, thickness=1)
    put_text(frame, f"Threshold: {int(back_threshold)}°", 
            (panel_x + 15, y_offset + 20), font_scale=0.5, color=COLORS["dark_gray"], thickness=1)
    
    # STATUS
    y_offset += line_height + 10
    status = "BAD POSTURE" if analysis["is_bad"] else "GOOD POSTURE"
    status_color = COLORS["red"] if analysis["is_bad"] else COLORS["green"]
    put_text(frame, f"Status: {status}", (panel_x + 15, y_offset), 
            font_scale=0.7, color=status_color, thickness=2)
    
    if analysis["primary_issue"]:
        issue_text = analysis["primary_issue"].replace("_", " ").upper()
        put_text(frame, f"Issue: {issue_text}", (panel_x + 15, y_offset + 25), 
                font_scale=0.6, color=COLORS["red"], thickness=2)

def draw_calibration_screen(frame, progress):
    """Draw calibration instructions"""
    h, w, _ = frame.shape
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    panel_w = 500
    panel_h = 300
    panel_x = (w - panel_w) // 2
    panel_y = (h - panel_h) // 2
    
    draw_modern_panel(frame, panel_x, panel_y, panel_w, panel_h,
                     color=COLORS["light_blue"], alpha=0.3, border_width=3)
    
    put_text(frame, "CALIBRATION MODE", (panel_x + 100, panel_y + 50),
            font_scale=1.0, color=COLORS["primary_blue"], thickness=2)
    
    put_text(frame, "Sit with good posture for", (panel_x + 80, panel_y + 100),
            font_scale=0.7, color=COLORS["dark_gray"], thickness=2)
    put_text(frame, "2 seconds to calibrate", (panel_x + 110, panel_y + 130),
            font_scale=0.7, color=COLORS["dark_gray"], thickness=2)
    
    bar_w = 300
    bar_h = 20
    bar_x = panel_x + (panel_w - bar_w) // 2
    bar_y = panel_y + 170
    
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), 
                 COLORS["light_gray"], -1)
    
    fill_w = int(bar_w * progress / 100)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                 COLORS["green"], -1)
    
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                 COLORS["primary_blue"], 2)
    
    put_text(frame, f"{progress}%", (bar_x + bar_w + 20, bar_y + bar_h),
            font_scale=0.7, color=COLORS["primary_blue"], thickness=2)
    
    if progress == 100:
        put_text(frame, "✓ CALIBRATION COMPLETE!", (panel_x + 80, panel_y + 250),
                font_scale=0.85, color=COLORS["green"], thickness=2)

def draw_break_screen(frame, timer):
    """Draw break time screen with stretches"""
    h, w, _ = frame.shape
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (100, 180, 255), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    panel_w = 600
    panel_h = 400
    panel_x = (w - panel_w) // 2
    panel_y = (h - panel_h) // 2
    
    draw_modern_panel(frame, panel_x, panel_y, panel_w, panel_h,
                     color=COLORS["warning_orange"], alpha=0.3, border_width=3)
    
    put_text(frame, "BREAK TIME!", (panel_x + 200, panel_y + 50),
            font_scale=1.2, color=COLORS["warning_orange"], thickness=3)
    
    remaining = timer.get_remaining_time()
    time_str = timer.format_time(remaining)
    put_text(frame, time_str, (panel_x + 180, panel_y + 120),
            font_scale=2.0, color=COLORS["white"], thickness=3)
    
    # Stretch tips
    put_text(frame, "Quick Tips:", (panel_x + 50, panel_y + 200),
            font_scale=0.8, color=COLORS["white"], thickness=2)
    
    tips = [
        "• Neck rolls: Rotate your neck gently",
        "• Shoulder shrugs: Lift shoulders to ears",
        "• Stand and stretch: Reach arms upward",
        "• Eye break: Look away from screen"
    ]
    
    y_pos = panel_y + 240
    for tip in tips:
        put_text(frame, tip, (panel_x + 50, y_pos),
                font_scale=0.6, color=COLORS["white"], thickness=1)
        y_pos += 30

def draw_skeleton_with_angles(frame, landmarks, mp_pose, analysis):
    """Draw skeleton with angle indicators"""
    h, w, _ = frame.shape
    
    connections = [
        (mp_pose.PoseLandmark.LEFT_EAR.value, mp_pose.PoseLandmark.LEFT_SHOULDER.value),
        (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value),
        (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value),
        (mp_pose.PoseLandmark.RIGHT_EAR.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_HIP.value),
        (mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value),
    ]
    
    for start, end in connections:
        start_lm = landmarks[start]
        end_lm = landmarks[end]
        
        start_pos = (int(start_lm.x * w), int(start_lm.y * h))
        end_pos = (int(end_lm.x * w), int(end_lm.y * h))
        
        color = COLORS["green"] if not analysis["is_bad"] else COLORS["red"]
        cv2.line(frame, start_pos, end_pos, color, 3)
    
    joint_indices = [
        mp_pose.PoseLandmark.LEFT_EAR.value,
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_HIP.value,
        mp_pose.PoseLandmark.LEFT_KNEE.value,
        mp_pose.PoseLandmark.RIGHT_EAR.value,
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_HIP.value,
        mp_pose.PoseLandmark.RIGHT_KNEE.value,
    ]
    
    for idx in joint_indices:
        lm = landmarks[idx]
        pos = (int(lm.x * w), int(lm.y * h))
        color = COLORS["primary_blue"]
        cv2.circle(frame, pos, 6, color, -1)
        cv2.circle(frame, pos, 6, COLORS["white"], 2)

def draw_help_overlay(frame):
    """Draw keyboard help overlay"""
    h, w, _ = frame.shape
    
    help_x = 20
    help_y = h - 150
    help_w = 300
    help_h = 130
    
    draw_modern_panel(frame, help_x, help_y, help_w, help_h,
                     color=COLORS["light_gray"], alpha=0.2, border_width=1)
    
    put_text(frame, "KEYBOARD CONTROLS", (help_x + 20, help_y + 20),
            font_scale=0.5, color=COLORS["dark_gray"], thickness=1)
    
    controls = [
        "C - Calibrate  S - Skip",
        "SPACE - Pause  Q - Quit",
        "H - Hide Help"
    ]
    
    y_pos = help_y + 45
    for control in controls:
        put_text(frame, control, (help_x + 15, y_pos),
                font_scale=0.45, color=COLORS["dark_gray"], thickness=1)
        y_pos += 25

# =============================
# MAIN APPLICATION
# =============================
def main():
    settings = load_settings()
    model_path = settings["model_path"]
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found: {model_path}")
        print("Download: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task")
        return
    
    # Setup MediaPipe
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
        print("Error: Webcam not found")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Initialize components
    posture_analyzer = AdvancedPostureAnalyzer(calibration_enabled=True)
    timer = TimerManager(settings["work_interval"], settings["break_duration"], 
                        sound_enabled=settings["sound_enabled"])
    
    cv2.namedWindow("PosturePal - Posture Detection")
    
    print("PosturePal Started!")
    print("Press 'C' to calibrate, 'Q' to quit, 'S' to skip calibration")
    print("Press 'SPACE' to pause/resume the timer")
    print("=" * 60)
    
    calibration_mode = True
    timestamp = 0
    fps_clock = time.time()
    fps_counter = 0
    current_fps = 0
    show_help = True
    
    with PoseLandmarker.create_from_options(options) as landmarker:
        mp_pose = mp.solutions.pose
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # FPS calculation
            fps_counter += 1
            if time.time() - fps_clock > 1:
                current_fps = fps_counter
                fps_counter = 0
                fps_clock = time.time()
            
            # Process frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect_for_video(mp_image, timestamp)
            timestamp += 33
            
            current_time = time.time()
            
            # Check timer phase transitions
            phase_status = timer.get_phase_status()
            
            # If calibration needed
            if calibration_mode and not posture_analyzer.calibration_data["calibration_complete"]:
                if result.pose_landmarks and len(result.pose_landmarks) > 0:
                    landmarks = result.pose_landmarks[0]
                    
                    try:
                        ear_l = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
                        shoulder_l = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                        hip_l = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                        knee_l = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                        elbow_l = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                        
                        neck_angle = calculate_angle(ear_l, shoulder_l, hip_l)
                        shoulder_angle = calculate_angle(ear_l, shoulder_l, elbow_l)
                        back_angle = calculate_back_angle(shoulder_l, hip_l, knee_l)
                        
                        cal_complete = posture_analyzer.calibrate(neck_angle, shoulder_angle, back_angle)
                        
                        if cal_complete:
                            print("\n✓ CALIBRATION COMPLETE!")
                            print(f"  Neck baseline: {posture_analyzer.calibration_data['neck_baseline']:.1f}°")
                            print(f"  Shoulder baseline: {posture_analyzer.calibration_data['shoulder_baseline']:.1f}°")
                            print(f"  Back baseline: {posture_analyzer.calibration_data['back_baseline']:.1f}°")
                            print("=" * 60)
                    
                    except Exception as e:
                        print(f"Calibration error: {e}")
                
                progress = posture_analyzer.get_calibration_progress()
                draw_calibration_screen(frame, progress)
            
            # Posture analysis
            else:
                # Draw break screen if in break mode
                if not timer.is_working:
                    draw_break_screen(frame, timer)
                
                # Draw posture analysis if in work mode or timer not paused
                elif result.pose_landmarks and len(result.pose_landmarks) > 0:
                    landmarks = result.pose_landmarks[0]
                    
                    try:
                        ear_l = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
                        shoulder_l = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                        hip_l = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                        knee_l = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                        elbow_l = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                        
                        neck_angle = calculate_angle(ear_l, shoulder_l, hip_l)
                        shoulder_angle = calculate_angle(ear_l, shoulder_l, elbow_l)
                        back_angle = calculate_back_angle(shoulder_l, hip_l, knee_l)
                        face_distance = calculate_face_distance(landmarks, mp_pose)
                        
                        analysis = posture_analyzer.analyze_posture(
                            neck_angle, shoulder_angle, back_angle, 
                            face_distance, settings, current_time
                        )
                        
                        # Send Windows notification if bad posture detected for 10+ seconds
                        if analysis["notification_triggered"]:
                            try:
                                toast = ToastNotifier()
                                issue_name = analysis["notification_issue"].replace("_", " ").title()
                                toast.show_toast("PosturePal Alert", 
                                               f"Bad posture detected: {issue_name}",
                                               duration=5,
                                               threaded=True)
                                print(f"Notification sent: {issue_name} detected for 10+ seconds")
                            except Exception as e:
                                print(f"Notification error: {e}")
                        
                        draw_posture_analysis_panel(frame, analysis, w, h)
                        draw_skeleton_with_angles(frame, landmarks, mp_pose, analysis)
                        
                        if analysis["is_bad"]:
                            warn_text = analysis["primary_issue"].replace("_", " ").upper()
                            warn_w = 350
                            warn_h = 60
                            warn_x = (w - warn_w) // 2
                            warn_y = h - 100
                            
                            draw_modern_panel(frame, warn_x, warn_y, warn_w, warn_h,
                                            color=COLORS["red"], alpha=0.2, border_width=2)
                            
                            put_text(frame, f"⚠ {warn_text}", 
                                    (warn_x + 50, warn_y + 40),
                                    font_scale=0.8, color=COLORS["red"], thickness=2)
                    
                    except Exception as e:
                        print(f"Analysis error: {e}")
            
            # Draw timer widget (always visible except calibration)
            if not calibration_mode:
                draw_timer_widget(frame, timer, w, h)
            
            # Draw FPS
            put_text(frame, f"FPS: {current_fps}", (w - 120, 30),
                    font_scale=0.6, color=COLORS["primary_blue"], thickness=2)
            
            # Draw help overlay
            if show_help:
                draw_help_overlay(frame)
            
            cv2.imshow("PosturePal - Posture Detection", frame)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('c') or key == ord('C'):
                print("\n🔄 Restarting calibration...")
                posture_analyzer.calibration_frames = 0
                posture_analyzer.calibration_data = {
                    "neck_baseline": None,
                    "shoulder_baseline": None,
                    "back_baseline": None,
                    "calibration_complete": False
                }
                calibration_mode = True
            elif key == ord('s') or key == ord('S'):
                print("\n⏭️  Skipping calibration, using default thresholds...")
                calibration_mode = False
            elif key == 32:  # SPACE
                timer.toggle_pause()
                state = "PAUSED" if timer.paused else "RESUMED"
                print(f"\n⏸️  Timer {state}")
            elif key == ord('h') or key == ord('H'):
                show_help = not show_help
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print summary
    print("\n" + "=" * 60)
    print("SESSION SUMMARY")
    print("=" * 60)
    print(f"Sessions Completed: {timer.sessions_completed}")
    print(f"Total Work Time: {timer.total_work_time/60:.1f}m")
    print(f"Total Break Time: {timer.total_break_time/60:.1f}m")
    print(f"Forward Head Time: {posture_analyzer.get_total_issue_time('forward_head')/60:.1f}m")
    print(f"Rounded Shoulders Time: {posture_analyzer.get_total_issue_time('rounded_shoulders')/60:.1f}m")
    print(f"Slouched Back Time: {posture_analyzer.get_total_issue_time('slouched_back')/60:.1f}m")
    print(f"Too Close Time: {posture_analyzer.get_total_issue_time('too_close')/60:.1f}m")
    print("=" * 60)

if __name__ == "__main__":
    main()
