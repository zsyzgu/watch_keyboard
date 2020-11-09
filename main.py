from finger_tracker import FingerTracker
import cv2

def input():
    frames = []

    for i in range(1,2):
        file_name = './data/1-1/L/' + str(i) + '.jpg'
        frame = cv2.imread(file_name)
        frames.append(frame)
    
    return frames

if __name__ == "__main__":
    frames = input()

    tracker = FingerTracker(1)

    for frame in frames:
        tracker.run(frame)
        output = tracker.output()
        cv2.imshow('i', output)
        cv2.waitKey(0)
