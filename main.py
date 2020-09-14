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

def deal_touch_data():
    tracker = FingerTracker(1)

    for r in range(3):
        for c in range(5):
            X = []
            Y = []
            for index in range(150):
                iteration = int(index // 15)
                row = int((index % 15) // 5)
                col = int(index % 5)
                if row == r and col == c:
                    file_name = 'raw/' + str(iteration) + '_' + str(row) + '_' + str(col) + '.jpg'
                    frame = cv2.imread(file_name)
                    tracker.run(frame)
                    output = tracker.output(str(index))
                    cv2.imshow('illustration', output)
                    cv2.waitKey(1)
                    for i in range(5):
                        if tracker.is_touch[i]:
                            x = tracker.endpoints[i][0]
                            y = tracker.palm_line#tracker.endpoints[i][1]
                            X.append(x)
                            Y.append(y)
            if len(X) > 0:
                plt.scatter(X,Y)
    plt.show()

def split_fingers(frame):
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

def try_ostu(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    thres, output = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    print(thres)
    return output

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
        cv2.imshow('e',output)
        key = cv2.waitKey(0)
        break

if __name__ == "__main__":
    #record_touch_data()
    #deal_touch_data()
    test()
