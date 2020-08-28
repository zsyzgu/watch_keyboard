import numpy as np
import cv2
from finger_tracker import FingerTracker
import time
import pickle
import img2video

if __name__ == "__main__":
    '''
    tracker = FingerTracker(1)

    for index in range(2000):
        frame = cv2.imread('raw/' + str(index) + '.jpg')
        tracker.run(frame)
        output = tracker.output(str(index))
        cv2.imshow('illustration', output)
        cv2.waitKey(1)
    '''
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    [map1, map2, _, _] = pickle.load(open('1.calib', 'rb'))

    tracker = FingerTracker(1)

    N = 2000
    output_arr = []
    image_arr = []

    for i in range(N):
        succ, frame = camera.read()
        if succ == False:
            break
        frame = cv2.remap(frame,map1,map2,cv2.INTER_LINEAR,borderValue=cv2.BORDER_CONSTANT)
        #frame = cv2.flip(frame, 1)
        tracker.run(frame)
        output = tracker.output(str(i))
        image = tracker.image
        cv2.imshow('illustration', output)
        cv2.waitKey(1)
        output_arr.append(output)
        image_arr.append(image)

    for i in range(N):
        print(i)
        cv2.imwrite('result/' + str(i) + '.jpg', output_arr[i])
        cv2.imwrite('raw/' + str(i) + '.jpg', image_arr[i])
    
    img2video.to_mp4(output_arr, 'result.mp4')
    