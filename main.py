import numpy as np
import cv2
from finger_tracker import FingerTracker
import time
import pickle
import img2video

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
    F = range(150,200)
    tracker = FingerTracker(1)
    time_log = []
    for index in F:
        frame = cv2.imread('raw/' + str(index) + '.jpg')
        t = time.clock()
        tracker.run(frame)
        time_log.append(time.clock()-t)
        output = tracker.output(str(index))
        cv2.waitKey(1)
        cv2.imshow('illustration', output)
    print('Mean time = %f' % (np.mean(time_log)))

if __name__ == "__main__":
    #record()
    test()
