import numpy as np
import cv2
from finger_tracker import FingerTracker
import time
import pickle
import img2video
import os
import matplotlib.pyplot as plt
import pickle

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
    r1 = cv2.erode(r1, kernel, iterations=4)

    sob2 = cv2.flip(cv2.Sobel(cv2.flip(frame, 1), -1, 1, 0, ksize=-1), 1)
    r2 = cv2.subtract(frame, sob2)
    _, r2 = cv2.threshold(r2, 55, 255, type=cv2.THRESH_TOZERO)
    r2 = cv2.erode(r2, kernel, iterations=4)

    r = cv2.max(r1,r2)
    return r

def test():
    input_folder = 'raw/'
    output_folder = 'fail_cases/'
    file_names = os.listdir(input_folder)
    tracker = FingerTracker(2)
    for index in range(len(file_names)):
        #file_name = file_names[index]
        file_name = '156.jpg'
        frame = cv2.imread(input_folder + file_name)
        t=time.clock()
        tracker.run(frame)
        output = tracker.output()
        t=time.clock()-t
        print(file_name)
        cv2.imshow('e',)
        key = cv2.waitKey(0)
        break

if __name__ == "__main__":
    #record()
    #test()
    pc = pickle.load(open('1.regist', 'rb'))
    
    row = [[], [], []]
    col = [[], []]
    L = np.min(pc[:,:,0])
    R = np.max(pc[:,:,0])
    D = np.min(pc[:,:,1])
    U = np.max(pc[:,:,1])
    for r in range(3):
        for c in range(5):
            x, y = pc[r,c]
            plt.scatter([x], [y])
            row[r].append(y)
            if c < 2:
                col[c].append(x)
    for r in range(3):
        row[r] = np.mean(row[r])
        plt.hlines(row[r], L, R)
    for c in range(2):
        col[c] = np.mean(col[c])
        plt.vlines(col[c], D, U)
    plt.show()
