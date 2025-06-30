# app.py - Flask Baby Monitor Web Application
from flask import Flask, render_template, Response, jsonify, request
import cv2
import json
import threading
import time
import logging
from datetime import datetime, timedelta
from collections import deque
import os

# Import your fall detection classes
from fall_detection import MotionDetector, FallDetector, AlertManager

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebBabyMonitor:
    def __init__(self):
        self.config = {
            'motion_threshold': 500,
            'fall_sensitivity': 0.6,
            'alert_cooldown': 10,
            'email_enabled': False,
            'sms_enabled': False,
            'sound_enabled': True
        }
        
        # Initialize detectors
        self.motion_detector = MotionDetector(
            threshold=self.config.get('motion_threshold', 500)
        )
        self.fall_detector = FallDetector(
            sensitivity=self.config.get('fall_sensitivity', 0.5),
            cooldown=self.config.get('alert_cooldown', 5)
        )
        self.alert_manager = AlertManager(self.config)
        
        # Video capture
        self.camera = cv2.VideoCapture(0)
        self.is_monitoring = False
        
        # Alert tracking
        self.current_alerts = []
        self.alert_history = deque(maxlen=100)
        self.stats = {
            'motion_events': 0,
            'fall_events': 0,
            'uptime': time.time(),
            'last_motion': None,
            'last_fall': None
        }
        
        # Thread safety
        self.lock = threading.Lock()
        
    def start_monitoring(self):
        """Start the monitoring process"""
        self.is_monitoring = True
        logger.info("Monitoring started")
        
    def stop_monitoring(self):
        """Stop the monitoring process"""
        self.is_monitoring = False
        logger.info("Monitoring stopped")
        
    def process_frame(self, frame):
        """Process a single frame for detection"""
        if not self.is_monitoring:
            return frame, {}
            
        results = {
            'motion': {'detected': False, 'confidence': 0},
            'fall': {'detected': False, 'confidence': 0},
            'timestamp': time.time()
        }
        
        try:
            # Motion detection
            motion_result = self.motion_detector.detect(frame)
            results['motion'] = motion_result
            
            # Fall detection (only if motion detected)
            if motion_result['detected']:
                fall_result = self.fall_detector.detect(frame)
                results['fall'] = fall_result
                
                # Process alerts
                self._process_detections(motion_result, fall_result)
                
                # Draw overlays
                frame = self._draw_overlays(frame, motion_result, fall_result)
                
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            
        return frame, results
        
    def _process_detections(self, motion_result, fall_result):
        """Process detection results and trigger alerts"""
        with self.lock:
            current_time = time.time()
            
            # Process fall detection
            if fall_result['detected']:
                self.stats['fall_events'] += 1
                self.stats['last_fall'] = current_time
                
                alert = {
                    'type': 'fall',
                    'confidence': fall_result['confidence'],
                    'timestamp': current_time,
                    'message': f"Fall detected with {fall_result['confidence']:.1%} confidence"
                }
                
                self.current_alerts.append(alert)
                self.alert_history.append(alert)
                self.alert_manager.trigger_alert('fall', fall_result)
                
            # Process motion detection
            elif motion_result['detected'] and motion_result['confidence'] > 0.3:
                self.stats['motion_events'] += 1
                self.stats['last_motion'] = current_time
                
                # Only alert for significant motion
                if motion_result['confidence'] > 0.6:
                    alert = {
                        'type': 'motion',
                        'confidence': motion_result['confidence'],
                        'timestamp': current_time,
                        'message': f"Significant motion detected ({motion_result['confidence']:.1%})"
                    }
                    
                    self.current_alerts.append(alert)
                    self.alert_history.append(alert)
                    
            # Clean old alerts (remove alerts older than 30 seconds)
            self.current_alerts = [
                alert for alert in self.current_alerts 
                if current_time - alert['timestamp'] < 30
            ]
    
    def _draw_overlays(self, frame, motion_result, fall_result):
        """Draw detection overlays on frame"""
        try:
            # Draw motion contours
            if motion_result.get('contours'):
                cv2.drawContours(frame, motion_result['contours'], -1, (0, 255, 0), 2)
            
            # Draw fall detection overlay
            if fall_result['detected']:
                cv2.rectangle(frame, (10, 10), (400, 80), (0, 0, 255), -1)
                cv2.putText(frame, "FALL DETECTED!", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Confidence: {fall_result['confidence']:.1%}", 
                           (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw motion indicator
            if motion_result['detected']:
                color = (0, 255, 255) if motion_result['confidence'] > 0.5 else (0, 255, 0)
                cv2.circle(frame, (frame.shape[1] - 30, 30), 10, color, -1)
            
            # Draw status text
            status_text = f"Motion: {motion_result['confidence']:.2f}"
            cv2.putText(frame, status_text, (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                       
        except Exception as e:
            logger.error(f"Error drawing overlays: {e}")
            
        return frame
    
    def get_alert_status(self):
        """Get current alert status"""
        with self.lock:
            return {
                'alerts': self.current_alerts[-5:],  # Last 5 alerts
                'stats': {
                    **self.stats,
                    'uptime_seconds': int(time.time() - self.stats['uptime']),
                    'is_monitoring': self.is_monitoring
                },
                'timestamp': time.time()
            }
    
    def get_settings(self):
        """Get current settings"""
        return self.config.copy()
        
    def update_settings(self, new_settings):
        """Update monitoring settings"""
        try:
            # Update config
            self.config.update(new_settings)
            
            # Update detector settings
            if 'motion_threshold' in new_settings:
                self.motion_detector.threshold = new_settings['motion_threshold']
            
            if 'fall_sensitivity' in new_settings:
                self.fall_detector.sensitivity = new_settings['fall_sensitivity']
                
            if 'alert_cooldown' in new_settings:
                self.fall_detector.cooldown = new_settings['alert_cooldown']
            
            logger.info(f"Settings updated: {new_settings}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating settings: {e}")
            return False

# Global monitor instance
monitor = WebBabyMonitor()

def generate_frames():
    """Generate video frames for streaming"""
    while True:
        try:
            success, frame = monitor.camera.read()
            if not success:
                logger.error("Failed to read camera frame")
                break
            
            # Process frame for detections
            frame, results = monitor.process_frame(frame)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            
            # Yield frame in multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
        except Exception as e:
            logger.error(f"Error generating frame: {e}")
            time.sleep(0.1)

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/alert_status')
def alert_status():
    """Get current alert status - THIS WAS MISSING"""
    try:
        status = monitor.get_alert_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting alert status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """Settings management"""
    if request.method == 'GET':
        return jsonify(monitor.get_settings())
    
    elif request.method == 'POST':
        try:
            new_settings = request.get_json()
            if monitor.update_settings(new_settings):
                return jsonify({'success': True, 'message': 'Settings updated'})
            else:
                return jsonify({'success': False, 'message': 'Failed to update settings'}), 400
        except Exception as e:
            logger.error(f"Error updating settings: {e}")
            return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/control/<action>')
def control(action):
    """Control monitoring (start/stop)"""
    try:
        if action == 'start':
            monitor.start_monitoring()
            return jsonify({'success': True, 'message': 'Monitoring started'})
        elif action == 'stop':
            monitor.stop_monitoring()
            return jsonify({'success': True, 'message': 'Monitoring stopped'})
        else:
            return jsonify({'success': False, 'message': 'Invalid action'}), 400
    except Exception as e:
        logger.error(f"Error controlling monitor: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/history')
def history():
    """Get alert history"""
    try:
        with monitor.lock:
            history_data = list(monitor.alert_history)
        
        # Convert timestamps to readable format
        for alert in history_data:
            alert['time_str'] = datetime.fromtimestamp(alert['timestamp']).strftime('%H:%M:%S')
            
        return jsonify({
            'history': history_data[-50:],  # Last 50 alerts
            'count': len(history_data)
        })
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'camera_connected': monitor.camera.isOpened(),
        'monitoring': monitor.is_monitoring
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("Baby Monitor starting...")
    
    # Start monitoring by default
    monitor.start_monitoring()
    
    # Get local IP for easier access
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    logger.info(f"Access URL: http://localhost:5000")
    logger.info(f"Network URL: http://{local_ip}:5000")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        monitor.stop_monitoring()
        if monitor.camera.isOpened():
            monitor.camera.release()
        cv2.destroyAllWindows()