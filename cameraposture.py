import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from datetime import datetime, timedelta
import json
import os
import subprocess
import platform

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
    "face_distance_threshold": 0.25,
    "model_path": "pose_landmarker_full.task",
    "show_landmarks": True,
    "minimal_mode": False,
    "sound_enabled": True
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

# -------------------------------
# CROSS-PLATFORM NOTIFICATIONS
# -------------------------------
class NotificationManager:
    def __init__(self):
        self.last_notification_time = 0
        self.notification_cooldown = 10
        self.system = platform.system()
    
    def send_notification(self, title, message):
        current_time = time.time()
        if current_time - self.last_notification_time >= self.notification_cooldown:
            try:
                if self.system == "Windows":
                    # Using PowerShell for Windows notifications
                    ps_script = f'''
                    [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
                    $template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
                    $toastXml = [xml] $template.GetXml()
                    $toastXml.GetElementsByTagName("text")[0].AppendChild($toastXml.CreateTextNode("{title}")) | Out-Null
                    $toastXml.GetElementsByTagName("text")[1].AppendChild($toastXml.CreateTextNode("{message}")) | Out-Null
                    $xml = New-Object Windows.Data.Xml.Dom.XmlDocument
                    $xml.LoadXml($toastXml.OuterXml)
                    $toast = [Windows.UI.Notifications.ToastNotification]::new($xml)
                    [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("PosturePal").Show($toast)
                    '''
                    subprocess.run(['powershell', '-Command', ps_script], 
                                 capture_output=True, timeout=2)
                elif self.system == "Darwin":  # macOS
                    subprocess.run(['osascript', '-e', 
                                  f'display notification "{message}" with title "{title}"'])
                elif self.system == "Linux":
                    subprocess.run(['notify-send', title, message])
                
                self.last_notification_time = current_time
            except Exception as e:
                print(f"Notification error: {e}")

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
# GLASSMORPHISM DRAWING UTILITIES
# -------------------------------
def draw_glass_panel(frame, x, y, width, height, alpha=0.3, blur_amount=15):
    """Draw a glassmorphism effect panel"""
    # Create ROI
    roi = frame[y:y+height, x:x+width].copy()
    
    # Apply blur
    blurred = cv2.GaussianBlur(roi, (blur_amount, blur_amount), 0)
    
    # Create white overlay with transparency
    overlay = np.ones_like(blurred) * 255
    overlay = cv2.addWeighted(blurred, 1-alpha, overlay, alpha, 0)
    
    # Add subtle gradient
    gradient = np.linspace(255, 230, height).reshape(-1, 1)
    gradient = np.repeat(gradient, width, axis=1)
    gradient = np.stack([gradient] * 3, axis=2).astype(np.uint8)
    overlay = cv2.addWeighted(overlay, 0.9, gradient, 0.1, 0)
    
    # Place back on frame
    frame[y:y+height, x:x+width] = overlay
    
    # Add border with orange tint
    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 200, 150), 2)
    
    # Add inner glow
    cv2.rectangle(frame, (x+1, y+1), (x + width-1, y + height-1), (255, 220, 180), 1)

def draw_progress_bar(frame, x, y, width, height, progress, color=(255, 140, 0)):
    """Draw an animated progress bar"""
    # Background
    cv2.rectangle(frame, (x, y), (x + width, y + height), (200, 200, 200), -1)
    
    # Progress fill
    fill_width = int(width * progress)
    if fill_width > 0:
        # Gradient effect
        for i in range(fill_width):
            alpha = 0.7 + 0.3 * (i / fill_width)
            color_adjusted = tuple(int(c * alpha) for c in color)
            cv2.line(frame, (x + i, y), (x + i, y + height), color_adjusted, 1)
    
    # Border
    cv2.rectangle(frame, (x, y), (x + width, y + height), (150, 150, 150), 1)

# -------------------------------
# ENHANCED SETTINGS UI
# -------------------------------
class SettingsUI:
    def __init__(self, settings):
        self.settings = settings.copy()
        self.show = False
        self.fields = [
            {"key": "work_interval", "name": "Work Interval", "type": "int", "unit": "min", 
             "divisor": 60, "min": 1, "max": 60, "step": 1},
            {"key": "break_duration", "name": "Break Duration", "type": "int", "unit": "sec", 
             "divisor": 1, "min": 10, "max": 120, "step": 5},
            {"key": "neck_angle_threshold", "name": "Neck Angle", "type": "int", "unit": "°", 
             "divisor": 1, "min": 140, "max": 180, "step": 5},
            {"key": "shoulder_slouch_threshold", "name": "Shoulder Angle", "type": "int", "unit": "°", 
             "divisor": 1, "min": 140, "max": 180, "step": 5},
            {"key": "posture_warning_duration", "name": "Warning Delay", "type": "int", "unit": "sec", 
             "divisor": 1, "min": 1, "max": 10, "step": 1},
            {"key": "face_distance_threshold", "name": "Distance Threshold", "type": "float", "unit": "", 
             "divisor": 1, "min": 0.15, "max": 0.40, "step": 0.01},
            {"key": "show_landmarks", "name": "Show Landmarks", "type": "bool", "unit": "", 
             "divisor": 1, "min": 0, "max": 1, "step": 1},
            {"key": "minimal_mode", "name": "Minimal Mode", "type": "bool", "unit": "", 
             "divisor": 1, "min": 0, "max": 1, "step": 1},
        ]
        self.hover_field = -1
        self.hover_button = None
    
    def toggle(self):
        self.show = not self.show
    
    def draw(self, frame):
        if not self.show:
            return
        
        h, w, _ = frame.shape
        
        panel_width = 600
        panel_height = 550
        panel_x = (w - panel_width) // 2
        panel_y = (h - panel_height) // 2
        
        # Draw glassmorphism panel
        draw_glass_panel(frame, panel_x, panel_y, panel_width, panel_height, alpha=0.25)
        
        # Title with orange gradient
        title_y = panel_y + 50
        cv2.putText(frame, "SETTINGS", (panel_x + 30, title_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 140, 0), 3)
        
        # Decorative line
        cv2.line(frame, (panel_x + 30, title_y + 10), 
                (panel_x + panel_width - 30, title_y + 10), (255, 180, 100), 2)
        
        # Fields
        y_offset = panel_y + 100
        field_height = 45
        
        for i, field in enumerate(self.fields):
            field_y = y_offset + i * field_height
            value = self.settings[field["key"]]
            
            # Hover effect
            if self.hover_field == i:
                cv2.rectangle(frame, (panel_x + 20, field_y - 25), 
                            (panel_x + panel_width - 20, field_y + 15), 
                            (255, 220, 180, 50), -1)
            
            # Field name
            cv2.putText(frame, field["name"], (panel_x + 30, field_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 2)
            
            if field["type"] == "bool":
                # Toggle switch
                switch_x = panel_x + panel_width - 100
                switch_y = field_y - 15
                switch_w = 60
                switch_h = 25
                
                # Switch background
                bg_color = (100, 200, 100) if value else (200, 100, 100)
                cv2.rectangle(frame, (switch_x, switch_y), 
                            (switch_x + switch_w, switch_y + switch_h), bg_color, -1)
                cv2.rectangle(frame, (switch_x, switch_y), 
                            (switch_x + switch_w, switch_y + switch_h), (150, 150, 150), 2)
                
                # Switch knob
                knob_x = switch_x + switch_w - 20 if value else switch_x + 5
                cv2.circle(frame, (knob_x + 10, switch_y + switch_h // 2), 10, (255, 255, 255), -1)
                
                # Store button position
                field["button_rect"] = (switch_x, switch_y, switch_w, switch_h)
            else:
                # Value display
                if field["type"] == "float":
                    display_value = f"{value / field['divisor']:.2f}"
                else:
                    display_value = f"{int(value / field['divisor'])}"
                
                value_str = f"{display_value} {field['unit']}"
                
                # Minus button
                minus_x = panel_x + panel_width - 200
                minus_y = field_y - 18
                minus_w = 35
                minus_h = 30
                
                minus_color = (255, 180, 100) if self.hover_button == (i, '-') else (220, 220, 220)
                cv2.rectangle(frame, (minus_x, minus_y), 
                            (minus_x + minus_w, minus_y + minus_h), minus_color, -1)
                cv2.rectangle(frame, (minus_x, minus_y), 
                            (minus_x + minus_w, minus_y + minus_h), (150, 150, 150), 2)
                cv2.putText(frame, "-", (minus_x + 12, minus_y + 22), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
                
                # Value
                value_x = minus_x + minus_w + 15
                cv2.putText(frame, value_str, (value_x, field_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 100, 0), 2)
                
                # Plus button
                plus_x = panel_x + panel_width - 70
                plus_y = field_y - 18
                plus_w = 35
                plus_h = 30
                
                plus_color = (255, 180, 100) if self.hover_button == (i, '+') else (220, 220, 220)
                cv2.rectangle(frame, (plus_x, plus_y), 
                            (plus_x + plus_w, plus_y + plus_h), plus_color, -1)
                cv2.rectangle(frame, (plus_x, plus_y), 
                            (plus_x + plus_w, plus_y + plus_h), (150, 150, 150), 2)
                cv2.putText(frame, "+", (plus_x + 10, plus_y + 22), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
                
                # Store button positions
                field["minus_rect"] = (minus_x, minus_y, minus_w, minus_h)
                field["plus_rect"] = (plus_x, plus_y, plus_w, plus_h)
        
        # Save and Cancel buttons
        button_y = panel_y + panel_height - 70
        
        # Save button
        save_x = panel_x + 120
        save_color = (100, 200, 100) if self.hover_button == 'save' else (150, 220, 150)
        cv2.rectangle(frame, (save_x, button_y), 
                     (save_x + 140, button_y + 45), save_color, -1)
        cv2.rectangle(frame, (save_x, button_y), 
                     (save_x + 140, button_y + 45), (100, 180, 100), 3)
        cv2.putText(frame, "SAVE", (save_x + 35, button_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Cancel button
        cancel_x = panel_x + 340
        cancel_color = (200, 100, 100) if self.hover_button == 'cancel' else (220, 150, 150)
        cv2.rectangle(frame, (cancel_x, button_y), 
                     (cancel_x + 140, button_y + 45), cancel_color, -1)
        cv2.rectangle(frame, (cancel_x, button_y), 
                     (cancel_x + 140, button_y + 45), (180, 100, 100), 3)
        cv2.putText(frame, "CANCEL", (cancel_x + 20, button_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Store button positions
        self.save_button_rect = (save_x, button_y, 140, 45)
        self.cancel_button_rect = (cancel_x, button_y, 140, 45)
        
        # Hint
        cv2.putText(frame, "Click buttons to adjust values | Press 'S' to toggle", 
                   (panel_x + 30, panel_y + panel_height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
    
    def handle_click(self, x, y):
        """Handle mouse clicks on settings UI"""
        if not self.show:
            return None
        
        # Check save button
        if hasattr(self, 'save_button_rect'):
            bx, by, bw, bh = self.save_button_rect
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return "save"
        
        # Check cancel button
        if hasattr(self, 'cancel_button_rect'):
            bx, by, bw, bh = self.cancel_button_rect
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return "cancel"
        
        # Check field buttons
        for i, field in enumerate(self.fields):
            if field["type"] == "bool":
                if "button_rect" in field:
                    bx, by, bw, bh = field["button_rect"]
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        self.settings[field["key"]] = not self.settings[field["key"]]
                        return "toggle"
            else:
                if "minus_rect" in field:
                    bx, by, bw, bh = field["minus_rect"]
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        self.adjust_value(i, False)
                        return "adjust"
                
                if "plus_rect" in field:
                    bx, by, bw, bh = field["plus_rect"]
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        self.adjust_value(i, True)
                        return "adjust"
        
        return None
    
    def handle_hover(self, x, y):
        """Handle mouse hover for visual feedback"""
        if not self.show:
            return
        
        self.hover_field = -1
        self.hover_button = None
        
        # Check save/cancel buttons
        if hasattr(self, 'save_button_rect'):
            bx, by, bw, bh = self.save_button_rect
            if bx <= x <= bx + bw and by <= y <= by + bh:
                self.hover_button = 'save'
        
        if hasattr(self, 'cancel_button_rect'):
            bx, by, bw, bh = self.cancel_button_rect
            if bx <= x <= bx + bw and by <= y <= by + bh:
                self.hover_button = 'cancel'
        
        # Check field buttons
        for i, field in enumerate(self.fields):
            if field["type"] != "bool":
                if "minus_rect" in field:
                    bx, by, bw, bh = field["minus_rect"]
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        self.hover_button = (i, '-')
                        self.hover_field = i
                
                if "plus_rect" in field:
                    bx, by, bw, bh = field["plus_rect"]
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        self.hover_button = (i, '+')
                        self.hover_field = i
    
    def adjust_value(self, field_index, increment):
        field = self.fields[field_index]
        current_value = self.settings[field["key"]]
        
        step = field["step"] * field["divisor"]
        new_value = current_value + (step if increment else -step)
        
        # Clamp to min/max
        min_val = field["min"] * field["divisor"]
        max_val = field["max"] * field["divisor"]
        self.settings[field["key"]] = max(min_val, min(max_val, new_value))

# -------------------------------
# ENHANCED COMPACT OVERLAY
# -------------------------------
class CompactOverlay:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 320
        self.height = 140
        self.dragging = False
        self.drag_offset_x = 0
        self.drag_offset_y = 0
        self.pulse_phase = 0
    
    def draw(self, frame, posture_tracker, timer, settings):
        if settings.get("minimal_mode", False):
            self.draw_minimal(frame, posture_tracker, timer)
            return
        
        # Glassmorphism panel
        draw_glass_panel(frame, self.x, self.y, self.width, self.height, alpha=0.2)
        
        # Pulse animation for warnings
        self.pulse_phase = (self.pulse_phase + 0.05) % (2 * np.pi)
        
        # Title
        cv2.putText(frame, "PosturePal", (self.x + 15, self.y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 140, 0), 2)
        
        # Bad posture time
        bad_minutes = posture_tracker.get_total_bad_posture_minutes()
        color = (100, 100, 255) if bad_minutes == 0 else (100, 100, 255)
        cv2.putText(frame, f"Bad Posture: {bad_minutes}m", 
                   (self.x + 15, self.y + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        
        # Timer status
        time_remaining = timer.get_time_remaining()
        time_str = timer.format_time(time_remaining)
        status = "Break Time!" if timer.on_break else "Next Break"
        status_color = (0, 200, 0) if timer.on_break else (255, 140, 0)
        
        cv2.putText(frame, f"{status}: {time_str}", 
                   (self.x + 15, self.y + 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)
        
        # Progress bar
        if not timer.on_break:
            progress = 1 - (time_remaining / timer.work_duration)
            draw_progress_bar(frame, self.x + 15, self.y + 100, 
                            self.width - 30, 12, progress, (255, 140, 0))
        
        # Play/Pause button
        button_x = self.x + 15
        button_y = self.y + 115
        button_size = 20
        
        button_color = (150, 220, 150) if timer.paused else (220, 200, 150)
        cv2.circle(frame, (button_x + button_size//2, button_y + button_size//2), 
                  button_size//2, button_color, -1)
        cv2.circle(frame, (button_x + button_size//2, button_y + button_size//2), 
                  button_size//2, (100, 100, 100), 2)
        
        if timer.paused:
            # Play triangle
            pts = np.array([
                [button_x + 7, button_y + 5],
                [button_x + 7, button_y + 15],
                [button_x + 15, button_y + 10]
            ], np.int32)
            cv2.fillPoly(frame, [pts], (80, 80, 80))
        else:
            # Pause bars
            cv2.rectangle(frame, (button_x + 6, button_y + 5), 
                         (button_x + 9, button_y + 15), (80, 80, 80), -1)
            cv2.rectangle(frame, (button_x + 11, button_y + 5), 
                         (button_x + 14, button_y + 15), (80, 80, 80), -1)
        
        # Settings button
        settings_x = button_x + 35
        cv2.circle(frame, (settings_x + 10, button_y + 10), 10, (220, 220, 220), -1)
        cv2.circle(frame, (settings_x + 10, button_y + 10), 10, (100, 100, 100), 2)
        cv2.putText(frame, "S", (settings_x + 6, button_y + 14), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 2)
        
        # Store button positions
        self.play_pause_rect = (button_x, button_y, button_size, button_size)
        self.settings_rect = (settings_x, button_y, 20, 20)
    
    def draw_minimal(self, frame, posture_tracker, timer):
        """Minimal mode - just a small status indicator"""
        self.width = 150
        self.height = 50
        
        draw_glass_panel(frame, self.x, self.y, self.width, self.height, alpha=0.15)
        
        bad_minutes = posture_tracker.get_total_bad_posture_minutes()
        time_str = timer.format_time(timer.get_time_remaining())
        
        cv2.putText(frame, f"Bad: {bad_minutes}m", (self.x + 10, self.y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)
        cv2.putText(frame, time_str, (self.x + 10, self.y + 42), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 140, 0), 1)
    
    def is_inside(self, x, y):
        return (self.x <= x <= self.x + self.width and 
                self.y <= y <= self.y + self.height)
    
    def is_play_pause_clicked(self, x, y):
        if hasattr(self, 'play_pause_rect'):
            bx, by, bw, bh = self.play_pause_rect
            return bx <= x <= bx + bw and by <= y <= by + bh
        return False
    
    def is_settings_clicked(self, x, y):
        if hasattr(self, 'settings_rect'):
            bx, by, bw, bh = self.settings_rect
            return bx <= x <= bx + bw and by <= y <= by + bh
        return False

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
            if self.pause_start_time:
                self.total_paused_time += time.time() - self.pause_start_time
            self.paused = False
        else:
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
    left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
    right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
    
    eye_distance = np.sqrt((left_eye.x - right_eye.x)**2 + (left_eye.y - right_eye.y)**2)
    return eye_distance

# -------------------------------
# Enhanced Break Reminder UI
# -------------------------------
def draw_break_reminder(frame):
    h, w, _ = frame.shape
    
    # Dark overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    
    # Center panel
    panel_width = 600
    panel_height = 400
    panel_x = (w - panel_width) // 2
    panel_y = (h - panel_height) // 2
    
    # Glassmorphism panel
    draw_glass_panel(frame, panel_x, panel_y, panel_width, panel_height, alpha=0.2)
    
    # Animated title
    pulse = int(10 * np.sin(time.time() * 3))
    cv2.putText(frame, "TIME FOR A BREAK!", 
               (panel_x + 80, panel_y + 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 140 + pulse, 0), 3)
    
    # Decorative line
    cv2.line(frame, (panel_x + 80, panel_y + 95), 
            (panel_x + panel_width - 80, panel_y + 95), (255, 180, 100), 3)
    
    messages = [
        ("20-20-20 Rule", 1.0, (255, 200, 100)),
        ("", 0.5, (255, 255, 255)),
        ("Look at something 20 feet away", 0.75, (240, 240, 240)),
        ("for 20 seconds", 0.75, (240, 240, 240)),
        ("", 0.5, (255, 255, 255)),
        ("This helps reduce eye strain", 0.6, (200, 200, 200)),
        ("and gives your posture a reset", 0.6, (200, 200, 200)),
    ]
    
    start_y = panel_y + 140
    for i, (msg, size, color) in enumerate(messages):
        if msg:
            text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, size, 2)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(frame, msg, (text_x, start_y + i * 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, size, color, 2)
    
    # Acknowledge button
    button_width = 250
    button_height = 50
    button_x = (w - button_width) // 2
    button_y = panel_y + panel_height - 90
    
    # Button with glow effect
    cv2.rectangle(frame, (button_x, button_y), 
                 (button_x + button_width, button_y + button_height), 
                 (255, 180, 100), -1)
    cv2.rectangle(frame, (button_x - 2, button_y - 2), 
                 (button_x + button_width + 2, button_y + button_height + 2), 
                 (255, 140, 0), 3)
    
    cv2.putText(frame, "ACKNOWLEDGE", 
               (button_x + 35, button_y + 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    # Hint text
    cv2.putText(frame, "Press SPACE or click button", 
               (panel_x + 160, panel_y + panel_height - 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

# -------------------------------
# Mouse Callback
# -------------------------------
def mouse_callback(event, x, y, flags, param):
    overlay, settings_ui, timer = param
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check settings UI clicks
        if settings_ui.show:
            action = settings_ui.handle_click(x, y)
            if action == "save":
                return  # Handled in main loop
            elif action == "cancel":
                return  # Handled in main loop
        else:
            # Check overlay clicks
            if overlay.is_settings_clicked(x, y):
                settings_ui.toggle()
            elif overlay.is_play_pause_clicked(x, y):
                timer.toggle_pause()
            elif overlay.is_inside(x, y):
                overlay.dragging = True
                overlay.drag_offset_x = x - overlay.x
                overlay.drag_offset_y = y - overlay.y
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if overlay.dragging:
            overlay.x = x - overlay.drag_offset_x
            overlay.y = y - overlay.drag_offset_y
        
        # Handle hover effects
        if settings_ui.show:
            settings_ui.handle_hover(x, y)
    
    elif event == cv2.EVENT_LBUTTONUP:
        overlay.dragging = False

# -------------------------------
# MAIN
# -------------------------------
def main():
    settings = load_settings()
    model_path = settings["model_path"]
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please download the pose landmarker model from:")
        print("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task")
        return
    
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
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Initialize components
    timer = WorkBreakTimer(settings["work_interval"], settings["break_duration"])
    posture_tracker = PostureTracker(settings["posture_warning_duration"])
    logger = PostureLogger(LOG_FILE)
    notif_manager = NotificationManager()
    settings_ui = SettingsUI(settings)
    compact_overlay = CompactOverlay(20, 20)
    
    # Window setup
    cv2.namedWindow("PosturePal - Desk Health Monitor")
    cv2.setMouseCallback("PosturePal - Desk Health Monitor", mouse_callback, 
                        (compact_overlay, settings_ui, timer))
    
    # Track various states
    window_focused = True
    last_focus_check = time.time()
    last_distance_warning_time = 0
    last_logged_posture_time = 0
    
    with PoseLandmarker.create_from_options(options) as landmarker:
        timestamp = 0
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS
        
        print("PosturePal started! Press 'Q' to quit, 'S' for settings.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)  # Mirror the frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            result = landmarker.detect_for_video(mp_image, timestamp)
            timestamp += 33
            
            h, w, _ = frame.shape
            is_bad_posture = False
            posture_type = None
            current_time = time.time()
            
            # Check window focus
            if current_time - last_focus_check > 1.0:
                try:
                    window_prop = cv2.getWindowProperty("PosturePal - Desk Health Monitor", 
                                                       cv2.WND_PROP_VISIBLE)
                    window_focused = window_prop >= 1
                except:
                    window_focused = False
                last_focus_check = current_time
            
            # Posture analysis
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
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
                    
                    # Distance warning with glassmorphism
                    if too_close and current_time - last_distance_warning_time > 5:
                        # Warning panel
                        warn_w = 450
                        warn_h = 80
                        warn_x = (w - warn_w) // 2
                        warn_y = h - 120
                        
                        draw_glass_panel(frame, warn_x, warn_y, warn_w, warn_h, alpha=0.2)
                        
                        pulse = int(15 * np.sin(current_time * 5))
                        cv2.putText(frame, "TOO CLOSE TO SCREEN!", 
                                   (warn_x + 50, warn_y + 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                                   (255, 165 + pulse, 0), 3)
                        
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
                            # Warning with glassmorphism
                            warn_w = 500
                            warn_h = 70
                            warn_x = (w - warn_w) // 2
                            warn_y = 50
                            
                            draw_glass_panel(frame, warn_x, warn_y, warn_w, warn_h, alpha=0.15)
                            
                            cv2.putText(frame, "HEAD FORWARD! Sit Up Straight!", 
                                       (warn_x + 40, warn_y + 45), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                                       (100, 100, 255), 3)
                            
                            if not window_focused:
                                notif_manager.send_notification("Posture Alert", 
                                                              "Head forward - sit up straight!")
                    
                    if shoulder_angle < settings["shoulder_slouch_threshold"]:
                        is_bad_posture = True
                        posture_type = "rounded_shoulders"
                        
                        if posture_tracker.should_show_warning():
                            warn_w = 520
                            warn_h = 70
                            warn_x = (w - warn_w) // 2
                            warn_y = 140
                            
                            draw_glass_panel(frame, warn_x, warn_y, warn_w, warn_h, alpha=0.15)
                            
                            cv2.putText(frame, "SHOULDERS ROUNDED! Roll them back!", 
                                       (warn_x + 30, warn_y + 45), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                                       (150, 100, 255), 3)
                            
                            if not window_focused:
                                notif_manager.send_notification("Posture Alert", 
                                                              "Rounded shoulders detected!")
                    
                    # Display angles with improved styling
                    if settings.get("show_landmarks", True):
                        shoulder_x = int(shoulder_l.x * w)
                        shoulder_y = int(shoulder_l.y * h)
                        
                        # Angle display with background
                        info_x = shoulder_x + 20
                        info_y = shoulder_y - 50
                        info_w = 130
                        info_h = 60
                        
                        # Small info panel
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (info_x, info_y), 
                                    (info_x + info_w, info_y + info_h), 
                                    (255, 255, 255), -1)
                        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                        cv2.rectangle(frame, (info_x, info_y), 
                                    (info_x + info_w, info_y + info_h), 
                                    (200, 200, 200), 2)
                        
                        neck_color = (0, 200, 0) if neck_angle >= settings["neck_angle_threshold"] else (0, 0, 255)
                        cv2.putText(frame, f'Neck: {int(neck_angle)}', 
                                   (info_x + 10, info_y + 25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, neck_color, 2)
                        
                        shoulder_color = (0, 200, 0) if shoulder_angle >= settings["shoulder_slouch_threshold"] else (0, 0, 255)
                        cv2.putText(frame, f'Shoulder: {int(shoulder_angle)}', 
                                   (info_x + 10, info_y + 48), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, shoulder_color, 2)
                        
                        # Draw pose landmarks
                        mp_drawing.draw_landmarks(
                            frame, result.pose_landmarks[0], POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(255, 180, 120), thickness=2, circle_radius=3),
                            mp_drawing.DrawingSpec(color=(255, 140, 0), thickness=3, circle_radius=2)
                        )
                
                except Exception as e:
                    print(f"Pose detection error: {e}")
            
            # Update posture tracker
            posture_tracker.update(is_bad_posture, posture_type)
            
            # Log bad posture events (every 30 seconds)
            if is_bad_posture and posture_tracker.should_show_warning():
                if current_time - last_logged_posture_time > 30:
                    duration = posture_tracker.get_bad_posture_duration()
                    logger.log_bad_posture(duration, posture_type or "unknown")
                    last_logged_posture_time = current_time
            
            # Check break timer
            timer_status = timer.check_timer()
            
            if timer_status in ["BREAK_TIME", "ON_BREAK"]:
                if not timer.break_acknowledged:
                    draw_break_reminder(frame)
                    if not window_focused and timer_status == "BREAK_TIME":
                        notif_manager.send_notification("Break Time!", 
                                                      "Time for your 20-20-20 break!")
            
            # Draw compact overlay (not during break reminder)
            if timer.break_acknowledged or timer_status not in ["BREAK_TIME", "ON_BREAK"]:
                compact_overlay.draw(frame, posture_tracker, timer, settings)
            
            # Draw settings UI
            settings_ui.draw(frame)
            
            cv2.imshow("PosturePal - Desk Health Monitor", frame)
            
            # Keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord(' '):
                if timer.on_break and not timer.break_acknowledged:
                    timer.acknowledge_break()
                    logger.log_break_taken()
            elif key == ord('s') or key == ord('S'):
                if settings_ui.show:
                    # Check if we should save
                    action = settings_ui.handle_click(0, 0)  # Dummy click to trigger save check
                    save_settings(settings_ui.settings)
                    settings = settings_ui.settings.copy()
                    timer.update_settings(settings["work_interval"], settings["break_duration"])
                    posture_tracker.warning_duration = settings["posture_warning_duration"]
                    print("Settings saved!")
                settings_ui.toggle()
            elif key == ord('m') or key == ord('M'):
                # Toggle minimal mode
                settings["minimal_mode"] = not settings.get("minimal_mode", False)
                save_settings(settings)
                print(f"Minimal mode: {'ON' if settings['minimal_mode'] else 'OFF'}")
            elif key == ord('l') or key == ord('L'):
                # Toggle landmarks
                settings["show_landmarks"] = not settings.get("show_landmarks", True)
                save_settings(settings)
                print(f"Landmarks: {'ON' if settings['show_landmarks'] else 'OFF'}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Log session end
    session_duration = int(time.time() - posture_tracker.last_check_time)
    logger.log_event("session_end", session_duration, "normal")
    print(f"\nSession ended. Total bad posture time: {posture_tracker.get_total_bad_posture_minutes()} minutes")

if __name__ == "__main__":
    main()

