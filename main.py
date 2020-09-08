import numpy as np
import cv2
from finger_tracker import FingerTracker
import time
import pickle
import img2video
import os

def record():
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

    tracker = FingerTracker(1)

    raws = []
    results = []
    for index in range(2000):
        succ, frame = camera.read()
        if succ == False:
            break
        tracker.run(frame)
        output = tracker.output(str(index))
        cv2.imshow('illustration', output)
        cv2.waitKey(1)
        raws.append(frame)
        results.append(output)
        
    for i in range(len(raws)):
        print(i)
        cv2.imwrite('raw/' + str(i) + '.jpg', raws[i])
    
    img2video.to_mp4(results, 'result.mp4')

def test():
    folder_name = 'raw/L/'
    file_names = os.listdir(folder_name)
    tracker = FingerTracker(1)
    for index in range(len(file_names)):
        #index = 0
        file_name = file_names[index]
        print(file_name)
        frame = cv2.imread(folder_name + file_name)
        tracker.run(frame)
        output = tracker.output(str(index))
        cv2.imshow('illustration', output)
        cv2.waitKey(1)

if __name__ == "__main__":
    #record()
    test()
