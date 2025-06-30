# fall_detection.py - Advanced Motion and Fall Detection
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time
import math
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
import threading
import os
import winsound  # For Windows sound alerts (use 'pygame' or 'playsound' for cross-platform)

logger = logging.getLogger(__name__)

class MotionDetector:
    def __init__(self, threshold=500, history_size=10):
        self.threshold = threshold
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=50
        )
        self.motion_history = deque(maxlen=history_size)
        self.first_frame = None
        self.frame_count = 0
        
    def detect(self, frame):
        """Detect motion in frame"""
        self.frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Initialize background
        if self.first_frame is None:
            self.first_frame = gray
            return {'detected': False, 'confidence': 0, 'area': 0, 'contours': []}
        
        # Background subtraction method
        fg_mask = self.background_subtractor.apply(frame)
        
        # Frame difference method (fallback)
        frame_delta = cv2.absdiff(self.first_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Combine both methods
        combined = cv2.bitwise_or(fg_mask, thresh)
        
        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        valid_contours = []
        total_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.threshold:
                valid_contours.append(contour)
                total_area += area
        
        # Calculate confidence based on motion area
        confidence = min(total_area / (frame.shape[0] * frame.shape[1]), 1.0)
        
        # Update motion history
        self.motion_history.append(confidence)
        
        # Update background periodically
        if self.frame_count % 30 == 0:
            self.first_frame = gray
        
        return {
            'detected': len(valid_contours) > 0,
            'confidence': confidence,
            'area': total_area,
            'contours': valid_contours
        }

class FallDetector:
    def __init__(self, sensitivity=0.5, cooldown=5):
        self.sensitivity = sensitivity
        self.cooldown = cooldown
        self.last_alert_time = 0
        
        # MediaPipe pose detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # History for pose analysis
        self.pose_history = deque(maxlen=30)  # 1 second at 30fps
        self.velocity_history = deque(maxlen=10)
        self.previous_pose = None
        
    def detect(self, frame):
        """Detect falls using pose estimation and motion analysis"""
        current_time = time.time()
        
        # Cooldown check
        if current_time - self.last_alert_time < self.cooldown:
            return {'detected': False, 'confidence': 0, 'position': None}
        
        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return {'detected': False, 'confidence': 0, 'position': None}
        
        # Extract key landmarks
        landmarks = results.pose_landmarks.landmark
        
        # Get key points
        head = landmarks[self.mp_pose.PoseLandmark.NOSE]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        
        # Calculate pose metrics
        pose_data = self._calculate_pose_metrics(
            head, left_shoulder, right_shoulder, 
            left_hip, right_hip, left_knee, right_knee
        )
        
        # Store pose history
        self.pose_history.append(pose_data)
        
        # Calculate velocity if we have previous pose
        if self.previous_pose:
            velocity = self._calculate_velocity(self.previous_pose, pose_data)
            self.velocity_history.append(velocity)
        
        self.previous_pose = pose_data
        
        # Analyze for fall patterns
        fall_confidence = self._analyze_fall_patterns()
        
        is_fall = fall_confidence > self.sensitivity
        
        if is_fall:
            self.last_alert_time = current_time
            logger.warning(f"Fall detected with confidence: {fall_confidence}")
        
        return {
            'detected': is_fall,
            'confidence': fall_confidence,
            'position': {
                'head': (head.x, head.y),
                'center': pose_data['center_of_mass']
            }
        }
    
    def _calculate_pose_metrics(self, head, l_shoulder, r_shoulder, l_hip, r_hip, l_knee, r_knee):
        """Calculate pose metrics for fall detection"""
        # Center of mass (approximation)
        center_x = (l_shoulder.x + r_shoulder.x + l_hip.x + r_hip.x) / 4
        center_y = (l_shoulder.y + r_shoulder.y + l_hip.y + r_hip.y) / 4
        
        # Body height (head to average of knees)
        knee_avg_y = (l_knee.y + r_knee.y) / 2
        body_height = abs(head.y - knee_avg_y)
        
        # Body width (shoulder to hip span)
        shoulder_width = abs(l_shoulder.x - r_shoulder.x)
        hip_width = abs(l_hip.x - r_hip.x)
        body_width = max(shoulder_width, hip_width)
        
        # Aspect ratio (width/height)
        aspect_ratio = body_width / max(body_height, 0.01)
        
        # Vertical alignment (head relative to body center)
        vertical_alignment = abs(head.x - center_x)
        
        # Posture angle (angle of spine approximation)
        spine_vector = np.array([center_x - head.x, center_y - head.y])
        vertical_vector = np.array([0, 1])
        
        if np.linalg.norm(spine_vector) > 0:
            cos_angle = np.dot(spine_vector, vertical_vector) / (
                np.linalg.norm(spine_vector) * np.linalg.norm(vertical_vector)
            )
            posture_angle = math.acos(np.clip(cos_angle, -1, 1))
        else:
            posture_angle = 0
        
        return {
            'center_of_mass': (center_x, center_y),
            'body_height': body_height,
            'body_width': body_width,
            'aspect_ratio': aspect_ratio,
            'vertical_alignment': vertical_alignment,
            'posture_angle': posture_angle,
            'head_position': (head.x, head.y)
        }
    
    def _calculate_velocity(self, prev_pose, curr_pose):
        """Calculate movement velocity between poses"""
        prev_center = prev_pose['center_of_mass']
        curr_center = curr_pose['center_of_mass']
        
        dx = curr_center[0] - prev_center[0]
        dy = curr_center[1] - prev_center[1]
        
        return math.sqrt(dx*dx + dy*dy)
    
    def _analyze_fall_patterns(self):
        """Analyze pose history for fall patterns"""
        if len(self.pose_history) < 10:
            return 0.0
        
        confidence_factors = []
        
        # 1. Sudden change in aspect ratio (body becomes more horizontal)
        recent_ratios = [pose['aspect_ratio'] for pose in list(self.pose_history)[-5:]]
        older_ratios = [pose['aspect_ratio'] for pose in list(self.pose_history)[-15:-10]]
        
        if len(older_ratios) > 0:
            ratio_change = np.mean(recent_ratios) - np.mean(older_ratios)
            if ratio_change > 0.3:  # Significant increase in aspect ratio
                confidence_factors.append(min(ratio_change, 1.0))
        
        # 2. Rapid vertical movement (falling motion)
        if len(self.velocity_history) >= 3:
            recent_velocities = list(self.velocity_history)[-3:]
            avg_velocity = np.mean(recent_velocities)
            if avg_velocity > 0.05:  # Significant movement
                confidence_factors.append(min(avg_velocity * 10, 1.0))
        
        # 3. Large posture angle change (body orientation)
        recent_angles = [pose['posture_angle'] for pose in list(self.pose_history)[-5:]]
        if len(recent_angles) > 0:
            max_angle = max(recent_angles)
            if max_angle > math.pi/3:  # More than 60 degrees from vertical
                confidence_factors.append(min(max_angle / (math.pi/2), 1.0))
        
        # 4. Head position relative to body center (head lower than expected)
        latest_pose = self.pose_history[-1]
        head_y = latest_pose['head_position'][1]
        center_y = latest_pose['center_of_mass'][1]
        
        if head_y > center_y + 0.1:  # Head significantly below center of mass
            confidence_factors.append(0.7)
        
        # 5. Sudden height reduction
        if len(self.pose_history) >= 10:
            recent_heights = [pose['body_height'] for pose in list(self.pose_history)[-5:]]
            older_heights = [pose['body_height'] for pose in list(self.pose_history)[-10:-5:]]
            
            if len(older_heights) > 0:
                height_change = np.mean(older_heights) - np.mean(recent_heights)
                if height_change > 0.1:  # Significant height reduction
                    confidence_factors.append(min(height_change * 5, 1.0))
        
        # Calculate overall confidence
        if not confidence_factors:
            return 0.0
        
        # Weight the factors (some are more important)
        weighted_confidence = np.mean(confidence_factors) * 0.8 + max(confidence_factors) * 0.2
        
        return min(weighted_confidence, 1.0)

class AlertManager:
    def __init__(self, config):
        self.config = config
        self.alert_history = deque(maxlen=100)
        
    def trigger_alert(self, alert_type, detection_data):
        """Trigger various types of alerts"""
        alert_info = {
            'type': alert_type,
            'timestamp': time.time(),
            'confidence': detection_data.get('confidence', 0),
            'data': detection_data
        }
        
        self.alert_history.append(alert_info)
        
        # Trigger different alert methods
        threading.Thread(target=self._send_alerts, args=(alert_info,), daemon=True).start()
    
    def _send_alerts(self, alert_info):
        """Send alerts through configured channels"""
        try:
            if self.config.get('email_enabled', False):
                self._send_email_alert(alert_info)
            
            if self.config.get('sms_enabled', False):
                self._send_sms_alert(alert_info)
            
            if self.config.get('sound_enabled', True):
                self._play_sound_alert(alert_info)
                
        except Exception as e:
            logger.error(f"Error sending alerts: {e}")
    
    def _send_email_alert(self, alert_info):
        """Send email alert"""
        try:
            # Email configuration (should be in config file)
            smtp_server = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
            smtp_port = int(os.environ.get('SMTP_PORT', '587'))
            email_user = os.environ.get('EMAIL_USER', '')
            email_pass = os.environ.get('EMAIL_PASS', '')
            recipient = os.environ.get('ALERT_EMAIL', '')
            
            if not all([email_user, email_pass, recipient]):
                logger.warning("Email configuration incomplete")
                return
            
            msg = MIMEMultipart()
            msg['From'] = email_user
            msg['To'] = recipient
            msg['Subject'] = f"Baby Monitor Alert - {alert_info['type'].title()}"
            
            body = f"""
            Alert Type: {alert_info['type'].title()}
            Confidence: {alert_info['confidence']:.2f}
            Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert_info['timestamp']))}
            
            Please check the baby monitor immediately.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(email_user, email_pass)
            server.send_message(msg)
            server.quit()
            
            logger.info("Email alert sent successfully")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def _send_sms_alert(self, alert_info):
        """Send SMS alert via Twilio"""
        try:
            # Twilio configuration
            account_sid = os.environ.get('TWILIO_ACCOUNT_SID', '')
            auth_token = os.environ.get('TWILIO_AUTH_TOKEN', '')
            from_phone = os.environ.get('TWILIO_PHONE', '')
            to_phone = os.environ.get('ALERT_PHONE', '')
            
            if not all([account_sid, auth_token, from_phone, to_phone]):
                logger.warning("SMS configuration incomplete")
                return
            
            # Twilio API endpoint
            url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
            
            message_body = f"ALERT: {alert_info['type'].title()} detected with {alert_info['confidence']:.1%} confidence at {time.strftime('%H:%M:%S')}"
            
            data = {
                'From': from_phone,
                'To': to_phone,
                'Body': message_body
            }
            
            response = requests.post(
                url,
                data=data,
                auth=(account_sid, auth_token)
            )
            
            if response.status_code == 201:
                logger.info("SMS alert sent successfully")
            else:
                logger.error(f"Failed to send SMS: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Failed to send SMS alert: {e}")
    
    def _play_sound_alert(self, alert_info):
        """Play sound alert"""
        try:
            # For Windows
            if os.name == 'nt':
                import winsound
                # Play system sound
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                # Or play custom sound file
                # winsound.PlaySound("alert.wav", winsound.SND_FILENAME)
            else:
                # For Linux/Mac - you can use pygame or other audio libraries
                # Example with system bell
                print('\a')  # Terminal bell
                
            logger.info("Sound alert played")
            
        except Exception as e:
            logger.error(f"Failed to play sound alert: {e}")

class BabyMonitor:
    def __init__(self, config=None):
        self.config = config or {}
        
        # Initialize detectors
        self.motion_detector = MotionDetector(
            threshold=self.config.get('motion_threshold', 500)
        )
        self.fall_detector = FallDetector(
            sensitivity=self.config.get('fall_sensitivity', 0.5),
            cooldown=self.config.get('alert_cooldown', 5)
        )
        
        # Initialize alert manager
        self.alert_manager = AlertManager(self.config)
        
        # Video capture
        self.cap = None
        self.is_running = False
        
    def start_monitoring(self, camera_index=0):
        """Start the monitoring system"""
        try:
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                raise ValueError(f"Could not open camera {camera_index}")
            
            self.is_running = True
            logger.info("Baby monitor started")
            
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to read frame")
                    break
                
                # Detect motion
                motion_result = self.motion_detector.detect(frame)
                
                # Detect falls (only if motion detected to save processing)
                fall_result = {'detected': False}
                if motion_result['detected']:
                    fall_result = self.fall_detector.detect(frame)
                
                # Trigger alerts if needed
                if fall_result['detected']:
                    self.alert_manager.trigger_alert('fall', fall_result)
                elif motion_result['detected'] and motion_result['confidence'] > 0.5:
                    self.alert_manager.trigger_alert('motion', motion_result)
                
                # Display frame (optional)
                if self.config.get('show_video', False):
                    self._draw_overlay(frame, motion_result, fall_result)
                    cv2.imshow('Baby Monitor', frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
        except Exception as e:
            logger.error(f"Error in monitoring: {e}")
        finally:
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Baby monitor stopped")
    
    def _draw_overlay(self, frame, motion_result, fall_result):
        """Draw detection overlays on frame"""
        # Draw motion contours
        if motion_result['contours']:
            cv2.drawContours(frame, motion_result['contours'], -1, (0, 255, 0), 2)
        
        # Draw status text
        status_text = f"Motion: {motion_result['confidence']:.2f}"
        if fall_result['detected']:
            status_text += f" | FALL DETECTED: {fall_result['confidence']:.2f}"
            cv2.putText(frame, "FALL DETECTED!", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.putText(frame, status_text, (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Configuration
    config = {
        'motion_threshold': 500,
        'fall_sensitivity': 0.6,
        'alert_cooldown': 10,
        'show_video': True,
        'email_enabled': False,  # Set to True and configure environment variables
        'sms_enabled': False,    # Set to True and configure Twilio
        'sound_enabled': True
    }
    
    # Create and start monitor
    monitor = BabyMonitor(config)
    
    try:
        monitor.start_monitoring(camera_index=0)
    except KeyboardInterrupt:
        print("\nStopping baby monitor...")
        monitor.stop_monitoring()