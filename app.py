from flask import Flask, render_template, Response, jsonify, request
import cv2
import threading
import time
import logging
from datetime import datetime
from collections import deque
from fall_detection import MotionDetector, FallDetector, AlertManager
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'c83e26f8536401cc60668254e2abeb9e45039c26c82e6969525c22ed3c93a73e'


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
        self.motion_detector = MotionDetector(self.config['motion_threshold'])
        self.fall_detector = FallDetector(self.config['fall_sensitivity'], self.config['alert_cooldown'])
        self.alert_manager = AlertManager(self.config)
        self.camera = cv2.VideoCapture(0)
        self.is_monitoring = False
        self.current_alerts = []
        self.alert_history = deque(maxlen=100)
        self.stats = {
            'motion_events': 0,
            'fall_events': 0,
            'uptime': time.time(),
            'last_motion': None,
            'last_fall': None
        }
        self.lock = threading.Lock()

    def start_monitoring(self):
        self.is_monitoring = True

    def stop_monitoring(self):
        self.is_monitoring = False

    def process_frame(self, frame):
        if not self.is_monitoring:
            return frame, {}

        results = {'motion': {'detected': False}, 'fall': {'detected': False}, 'timestamp': time.time()}
        try:
            motion_result = self.motion_detector.detect(frame)
            results['motion'] = motion_result

            if motion_result['detected']:
                fall_result = self.fall_detector.detect(frame)
                results['fall'] = fall_result
                self._process_detections(motion_result, fall_result)
                frame = self._draw_overlays(frame, motion_result, fall_result)
        except Exception as e:
            logger.error(f"Processing error: {e}")
        return frame, results

    def _process_detections(self, motion_result, fall_result):
        with self.lock:
            now = time.time()
            if fall_result['detected']:
                self.stats['fall_events'] += 1
                self.stats['last_fall'] = now
                alert = {
                    'type': 'fall',
                    'confidence': fall_result['confidence'],
                    'timestamp': now,
                    'message': f"Fall detected ({fall_result['confidence']:.1%})"
                }
                self.current_alerts.append(alert)
                self.alert_history.append(alert)
                self.alert_manager.trigger_alert('fall', fall_result)

            elif motion_result['detected'] and motion_result['confidence'] > 0.6:
                self.stats['motion_events'] += 1
                self.stats['last_motion'] = now
                alert = {
                    'type': 'motion',
                    'confidence': motion_result['confidence'],
                    'timestamp': now,
                    'message': f"Motion detected ({motion_result['confidence']:.1%})"
                }
                self.current_alerts.append(alert)
                self.alert_history.append(alert)

            self.current_alerts = [a for a in self.current_alerts if now - a['timestamp'] < 30]

    def _draw_overlays(self, frame, motion_result, fall_result):
        if motion_result.get('contours'):
            cv2.drawContours(frame, motion_result['contours'], -1, (0, 255, 0), 2)
        if fall_result['detected']:
            cv2.putText(frame, "FALL DETECTED!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame

    def get_alert_status(self):
        with self.lock:
            return {
                'alerts': self.current_alerts[-5:],
                'stats': {
                    **self.stats,
                    'uptime_seconds': int(time.time() - self.stats['uptime']),
                    'is_monitoring': self.is_monitoring
                },
                'timestamp': time.time()
            }


monitor = WebBabyMonitor()


def generate_frames():
    while True:
        success, frame = monitor.camera.read()
        if not success:
            break
        frame, _ = monitor.process_frame(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/')
def index():
    return "Baby Monitor Running"


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/alert_status')
def alert_status():
    return jsonify(monitor.get_alert_status())


@app.route('/control/<action>')
def control(action):
    if action == 'start':
        monitor.start_monitoring()
        return jsonify({"started": True})
    elif action == 'stop':
        monitor.stop_monitoring()
        return jsonify({"stopped": True})
    return jsonify({"error": "Invalid action"}), 400


if __name__ == '__main__':
    monitor.start_monitoring()
    app.run(host='0.0.0.0', port=5000)
