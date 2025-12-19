import cv2
import numpy as np
import mediapipe as mp

class HandDetector:
    def __init__(self):
        self.detector = mp.solutions.hands.Hands(
            static_image_mode=False, 
            max_num_hands=1,
            min_detection_confidence=0.7
        )

    def enhance_for_asl(self, image_rgb):
        """Improve contrast to distinguish overlapping fingers (M/N/T/S)"""
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

    def get_hand_crop(self, frame):
        enhanced = self.enhance_for_asl(frame)
        results = self.detector.process(enhanced)
        if results.multi_hand_landmarks:
            # Logic to calculate BBox and crop goes here (from app.py)
            return enhanced, results.multi_hand_landmarks[0]
        return None, None