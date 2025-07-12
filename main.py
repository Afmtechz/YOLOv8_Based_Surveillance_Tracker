import os
import sys
import cv2
from ultralytics import YOLO
import threading
import pygame

pygame.mixer.init()

# Load the model
yolo = YOLO('yolov8s.pt')

class detect:
    def __init__(self, model):
        # Start webcam
        videoCap = cv2.VideoCapture(0)

        # Helper to get correct path whether running as script or .exe
        def resource_path(relative_path):
            try:
                # PyInstaller creates a temp folder and stores path in _MEIPASS
                base_path = sys._MEIPASS
            except AttributeError:
                base_path = os.path.abspath(".")

            return os.path.join(base_path, relative_path)

        # Sound play flag to avoid playing sound too frequently
        sound_playing = False

        # Function to play sound in a separate thread
        def play_alert():
            global sound_playing
            sound_playing = True
            sound_file = resource_path("alert.mp3")  # or "alert.wav"
            pygame.mixer.music.load(sound_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                continue
            sound_playing = False

        # Function to assign color to each class
        def getColours(cls_num):
            base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            color_index = cls_num % len(base_colors)
            increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
            color = [
                (base_colors[color_index][i] + increments[color_index][i] *
                (cls_num // len(base_colors))) % 256 for i in range(3)
            ]
            return tuple(color)

        # Main loop
        while True:
            ret, frame = videoCap.read()
            if not ret:
                continue

            results = yolo.track(frame, stream=True)

            for result in results:
                classes_names = result.names

                for box in result.boxes:
                    if box.conf[0] > 0.4:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls = int(box.cls[0])
                        class_name = classes_names[cls]
                        colour = getColours(cls)

                        # Draw bounding box and label
                        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                        cv2.putText(frame, f'{class_name} {box.conf[0]:.2f}', (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)

                        # If person is detected and not already playing sound, trigger it
                        if class_name.lower() == 'person' and not sound_playing:
                            threading.Thread(target=play_alert).start()

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        videoCap.release()
        cv2.destroyAllWindows()
if __name__ == "__main__":
    detect(yolo)