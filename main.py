import numpy as np
import cv2
from finger_tracker import FingerTracker
import time

if __name__ == "__main__":
    camera = cv2.VideoCapture('1.mp4')

    tracker = FingerTracker(1)

    while True:
        succ, frame = camera.read()
        if succ == False:
            break
        frame = cv2.flip(frame, 1)
        tracker.run(frame)
        output = tracker.output()
        cv2.imshow('illustration', output)
        cv2.waitKey(1)
