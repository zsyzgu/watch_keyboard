import numpy as np
import cv2
from finger_tracker import FingerTracker
import time
import pickle
import img2video
import os
import matplotlib.pyplot as plt
import pickle
import scipy.signal as signal

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

def erode_fingers(frame):
    kernel = np.uint8(np. ones((1, 3)))

    sob1 = cv2.Sobel(frame, -1, 1, 0, ksize=-1)
    r1 = cv2.subtract(frame, sob1)
    _, r1 = cv2.threshold(r1, 55, 255, type=cv2.THRESH_TOZERO)
    r1 = cv2.erode(r1, kernel, iterations=2)

    sob2 = cv2.flip(cv2.Sobel(cv2.flip(frame, 1), -1, 1, 0, ksize=-1), 1)
    r2 = cv2.subtract(frame, sob2)
    _, r2 = cv2.threshold(r2, 55, 255, type=cv2.THRESH_TOZERO)
    r2 = cv2.erode(r2, kernel, iterations=2)

    r = cv2.max(r1,r2)
    return r

'''
def test():
    tracker = FingerTracker(2)
    files = os.listdir('raw/')

    ts = []
    for file in files:
        frame = cv2.imread('raw/' + file)
        t=time.clock()
        tracker.run(frame)
        output = tracker.output()
        t=time.clock()-t
        ts.append(t)
        print(t)
        cv2.imshow('e',output)
        cv2.waitKey(1)
    print(np.mean(ts))
'''

def test():
    tracker_L = FingerTracker(1)
    tracker_R = FingerTracker(2)

    '''
    files = open('debug.txt').readlines()
    for file_name in files:
        file_name = file_name.strip()
        print(file_name)
        
        frame_L = cv2.imread(file_name)
        tracker_L.run(frame_L)
        output_L = tracker_L.output()
        cv2.imshow('illustration', output_L)
        cv2.waitKey(0)
    '''

    folder_name = 'data/2-1/'
    for i in range(1000):
        file_L = folder_name + 'L/' + str(i) + '.jpg'
        file_R = folder_name + 'R/' + str(i) + '.jpg'
        if os.path.exists(file_L) == False:
            break
        frame_L = cv2.imread(file_L)
        frame_R = cv2.imread(file_R)
        tracker_L.run(frame_L)
        tracker_R.run(frame_R)
        output_L = tracker_L.output()
        output_R = tracker_R.output()
        output = np.hstack([output_L, output_R])
        cv2.imshow('illustration', output)
        if np.any(np.array(tracker_R.fingertips)[:,0] == -1):
            print(file_R)
            cv2.waitKey(0)
        else:
            cv2.waitKey(1)



if __name__ == "__main__":
    #record()
    test()
