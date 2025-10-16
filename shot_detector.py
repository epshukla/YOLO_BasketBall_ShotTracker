from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
from utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos
import time


class ShotDetector:
    def __init__(self):
        self.model = YOLO("/home/theperson/Vajra/MyProjects/Basketball_ Tracking/AI-Basketball-Shot-Detection-Tracker-master/ball_hoop/v8/best (1).pt")
        self.class_names = ['Basketball', 'Basketball Hoop']

        #self.cap = cv2.VideoCapture("video_test_5.mp4")
        self.cap = cv2.VideoCapture(0)

        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = f"output_{timestamp}.mp4"
        self.out = cv2.VideoWriter(output_filename, fourcc, self.fps, (self.width, self.height))

        self.ball_pos = []  
        self.hoop_pos = []  
        self.frame_count = 0
        self.frame = None

        self.makes = 0
        self.attempts = 0

        # Used to detect shots (upper and lower region)
        self.up = False
        self.down = False
        self.up_frame = 0
        self.down_frame = 0

        # Used for green and red overlay color after make/miss
        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)

        print(f" Output video will be saved as: {output_filename}")
        self.run()

    def run(self):
        while True:
            ret, self.frame = self.cap.read()
            if not ret:
                print(" End of video or read error.")
                break

            results = self.model(self.frame, stream=True)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1

                    # Confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100

                    # Class Name
                    cls = int(box.cls[0])
                    current_class = self.class_names[cls]
                    center = (int(x1 + w / 2), int(y1 + h / 2))

                    if (conf > .3 or (in_hoop_region(center, self.hoop_pos) and conf > 0.15)) and current_class == "Basketball":
                        self.ball_pos.append((center, self.frame_count, w, h, conf))
                        cvzone.cornerRect(self.frame, (x1, y1, w, h))

                    if conf > .5 and current_class == "Basketball Hoop":
                        self.hoop_pos.append((center, self.frame_count, w, h, conf))
                        cvzone.cornerRect(self.frame, (x1, y1, w, h))

            self.clean_motion()
            self.shot_detection()
            self.display_score()
            self.frame_count += 1

            self.out.write(self.frame)

            cv2.imshow('Basketball Shot Detector', self.frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(" Interrupted by user.")
                break

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        print("🎬 Video saved successfully.")

    def clean_motion(self):
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)
        for i in range(len(self.ball_pos)):
            cv2.circle(self.frame, self.ball_pos[i][0], 2, (0, 0, 255), 2)

        if len(self.hoop_pos) > 1:
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)
            cv2.circle(self.frame, self.hoop_pos[-1][0], 2, (128, 128, 0), 2)

    def shot_detection(self):
        if len(self.hoop_pos) > 0 and len(self.ball_pos) > 0:
            if not self.up:
                self.up = detect_up(self.ball_pos, self.hoop_pos)
                if self.up:
                    self.up_frame = self.ball_pos[-1][1]

            if self.up and not self.down:
                self.down = detect_down(self.ball_pos, self.hoop_pos)
                if self.down:
                    self.down_frame = self.ball_pos[-1][1]

            if self.frame_count % 10 == 0:
                if self.up and self.down and self.up_frame < self.down_frame:
                    self.attempts += 1
                    self.up = False
                    self.down = False

                    if score(self.ball_pos, self.hoop_pos):
                        self.makes += 1
                        self.overlay_color = (0, 255, 0)
                        self.fade_counter = self.fade_frames
                    else:
                        self.overlay_color = (0, 0, 255)
                        self.fade_counter = self.fade_frames

    def display_score(self):
        text = f"{self.makes} / {self.attempts}"
        cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
        cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)

        if self.fade_counter > 0:
            alpha = 0.2 * (self.fade_counter / self.fade_frames)
            overlay = np.full_like(self.frame, self.overlay_color, dtype=np.uint8)
            self.frame = cv2.addWeighted(self.frame, 1 - alpha, overlay, alpha, 0)
            self.fade_counter -= 1


if __name__ == "__main__":
    ShotDetector()
