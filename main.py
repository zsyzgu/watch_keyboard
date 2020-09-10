import numpy as np
import cv2
from finger_tracker import FingerTracker
import time
import pickle
import img2video
import os
import matplotlib.pyplot as plt

def record():
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

    tracker = FingerTracker(1)
    
    results = []
    raws = []
    for index in range(300):
        succ, frame = camera.read()
        if succ == False:
            break
        tracker.run(frame)
        output = tracker.output(str(index))
        cv2.imshow('illustration', output)
        cv2.waitKey(1)

        raws.append(frame)
        results.append(output)
    
    for index in range(len(raws)):
        cv2.imwrite('raw/' + str(index) + '.jpg', raws[index])
    img2video.to_mp4(results)

def record_touch_data():
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

    tracker = FingerTracker(1)

    index = 0

    while True:
        succ, frame = camera.read()
        if succ == False:
            break
        tracker.run(frame)
        output = tracker.output(str(index))
        cv2.imshow('illustration', output)
        cv2.waitKey(1)

        iteration = int(index // 15)
        row = int((index % 15) // 5)
        col = int(index % 5)
        print('[%d] r = %d, c = %d' % (iteration, row, col))
        finger = col
        if finger == 0:
            finger = 1

        correct = tracker.is_touch_down[finger]
        if correct:
            cv2.imwrite('raw/' + str(iteration) + '_' + str(row) + '_' + str(col) + '.jpg', frame)
            index += 1
            if index == 150:
                break

def split_fingers(frame):
    frame=cv2.UMat(frame)
    kernel = np.uint8(np.ones((1, 3)))
    B=cv2.Sobel(frame,-1,1,0,ksize=-1)
    r1=cv2.subtract(frame,cv2.min(frame,B))
    _, r1 = cv2.threshold(r1, 55, 255, type=cv2.THRESH_TOZERO)
    r1 = cv2.erode(r1, kernel, iterations=4)
    C=cv2.flip(cv2.Sobel(cv2.flip(frame,1),-1,1,0,ksize=-1),1)
    r2 = cv2.subtract(frame,cv2.min(frame,C))
    _, r2 = cv2.threshold(r2, 55, 255, type=cv2.THRESH_TOZERO)
    r2 = cv2.erode(r2, kernel, iterations=4)
    r = cv2.max(r1,r2)
    return r

def test():
    folder_name = 'raw/'
    file_names = os.listdir(folder_name)
    tracker = FingerTracker(1)
    for index in range(0,250):
        file_name = file_names[index]
        print(file_name)
        
        frame = cv2.imread(folder_name + file_name)
        tracker.run(frame)
        output = tracker.output()

        cv2.imshow('illustration', output)
        cv2.waitKey(0)
        break

if __name__ == "__main__":
    #record()
    test()
