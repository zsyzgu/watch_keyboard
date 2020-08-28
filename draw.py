import numpy as np
import matplotlib.pyplot as plt
from finger_tracker import FingerTracker
import time
import pickle
import img2video
import cv2

if __name__ == "__main__":
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    [map1, map2, roi, mtx] = pickle.load(open('1.calib', 'rb'))

    image = np.zeros((500,500))
    tracker = FingerTracker(1)

    while True:
        succ, frame = camera.read()
        if succ == False:
            break
        tracker.run(frame)
        output = tracker.output()
        cv2.imshow('i',output)

        for i in range(4):
            if tracker.is_touch[i]:
                [x,y] = tracker.endpoints[i]
                print([x,y])
                x = int((x+5) * 50)
                y = int((15-y) * 50)
                cv2.circle(image, tuple([x,y]), 2, 255, cv2.FILLED)
                cv2.imshow('e',image)
        if tracker.is_touch[4]:
            image = np.zeros((500,500))
            cv2.imshow('e',image)

        cv2.waitKey(1)
