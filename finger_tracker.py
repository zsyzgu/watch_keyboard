import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import io
import math
from scipy.stats import kurtosis

class FingerTracker: # 1280 * 960
    CONTOURS_MIN_AREA = 6000 # the minimun area of contours of a finger
    PALM_THRESHOLD = 400 # the upper rows belong to the palm
    FINGER_GAP = 50 # The minimum distance between two finger in x-axis
    FIXED_BRIGHTNESS = True

    def __init__(self, camera_id):
        self.init_fingertips()
        self.BRIGHTNESS_THRESHOLD = 100 # Default brightness threshold
        self.init_camera_para(camera_id)
    
    def init_fingertips(self):
        self.THUMB_R_EDGE = 300 # the left cols just belong to the thumb
        self.TIP_HEIGHT = 12 # Deal with two recognized points on a fingertip
        self.TIP_WIDTH = 100 # Max width of a tip (excluding thumb)
        self.THUMB_WIDTH = 300 # Max width of a thumb
        self.SECOND_ROW_DELTA = 60 # The second row of the keyboard is x pixels lower than camera_cy
        self.MOVEMENT_THRESHOLD = 80 # The maximun moving pixels of a fintertip in one frame
        self.MIN_PRESSURE = 0 # Min pressure in a contact
        self.MIN_ENDPOINT_BRIGHTNESS = 0.7 # The endpoint should be bright enough when contacting
        self.finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinkie']
        self.fingertips = [[-1, -1]] * 5
        self.endpoints = [[-1,-1]] * 5 # endpoint on the desk calned by touchpoint
        self.is_touch = [False for i in range(5)]
        self.is_touch_down = [False for i in range(5)]
        self.is_touch_up = [False for i in range(5)]
        self.curr_middle_finger = [-1, -1] # The tip of the middle finger in the latest available frame
        self.palm_line = 0 # The root of the middle finger (in y axis)

    def init_camera_para(self, camera_id):
        self.camera_id = camera_id
        self.camera_H = 0.75
        if camera_id == 1:
            self.fx = 411.58
            self.fy = 412.05
            self.cx = 637.29
            self.cy = 463.37
        elif camera_id == 2:
            self.fx = 391.87
            self.fy = 393.57
            self.cx = 673.29
            self.cy = 452.71
        else:
            print('Camera ID Error.')

    def find_contours(self, frame, threshold, hier):
        _, binary = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, hier, cv2.CHAIN_APPROX_NONE)
        return contours

    def find_table(self, gray, brightness_threshold): # The table is the max contour connect with the bottom
        contours = self.find_contours(gray, brightness_threshold, cv2.RETR_EXTERNAL)
        bottoms = []
        for contour in contours:
            if np.max(contour[:, :, 1]) == self.N-1:
                bottoms.append(contour)
        if len(bottoms) == 0:
            return []
        table_contour = max(bottoms, key=cv2.contourArea)
        return table_contour

    def find_fingers(self, gray, brightness_threshold): # Find chunks containing fingers
        contours = self.find_contours(gray, brightness_threshold, cv2.RETR_EXTERNAL)

        fingers = []
        for contour in contours:
            if cv2.contourArea(contour) >= self.CONTOURS_MIN_AREA: # The finger must have enough area
                points = contour[:,0]
                fingers.append(np.array(points))

        return fingers

    def find_brightness_threshold(self, image):
        if self.FIXED_BRIGHTNESS:
            return 55

        palm_image = image[:self.N//2,:]

        hist = cv2.calcHist([palm_image],[0],None,[256],[0,256])
        peak_id = np.argmax(hist[:self.BRIGHTNESS_THRESHOLD])
        valley_id = np.argmin(hist[peak_id:self.BRIGHTNESS_THRESHOLD]) + peak_id

        for i in range(valley_id, 255):
            if hist[i] >= hist[valley_id] * 1.1:
                return i

        return self.BRIGHTNESS_THRESHOLD

    def find_palm_line(self, image): # palm_line = the root of the middel finger
        if self.fingertips[2][0] != -1:
            self.curr_middle_finger = self.fingertips[2]
        [x, y] = self.curr_middle_finger # The tip of the middle finger

        L_COL = max(x, int(0.5 * self.M))
        R_COL = int(0.8 * self.M)
        CHUNK = 30
        height = 0 # Find the lowest black pixel between fingers
        chunk = -1
        for i in range(100, int(self.cy), CHUNK):
            if np.count_nonzero(image[i:i+CHUNK,L_COL:R_COL]) != (R_COL-L_COL)*CHUNK:
                chunk = i
                break
        if chunk != -1:
            for i in range(chunk, int(self.cy)):
                if np.count_nonzero(image[i,L_COL:R_COL]) != R_COL-L_COL:
                    height = i
                    break
        self.palm_height = height

        palm_line = self.palm_line
        
        x = int((x + self.M//2) // 2)
        if x != -1:
            arr = image[:,x].copy()
            lb = max(height - 200, 50)
            ub = height + 20
            max_diff = 0
            peak = arr[lb]
            for i in range(lb,ub):
                if arr[i] > arr[i-1]:
                    peak = arr[i]
                elif peak - arr[i] > max_diff:
                    max_diff = peak - arr[i]
                    palm_line = i

        return palm_line

    def find_fingertips(self, image, brightness_threshold):
        fingers = self.find_fingers(image, brightness_threshold)
        fingertips = []
        self.has_thumb = False

        for finger in fingers:
            l = np.argmin(finger[:,0])
            r = np.argmax(finger[:,0])
            X = finger[l: r + 1, 0]
            Y = finger[l: r + 1, 1]
            
            ids = []
            radius = self.FINGER_GAP // 2
            point_num = len(X)

            for i in range(radius + 1, point_num - radius - 1): # Find peaks
                if (Y[i] > Y[i - 1] and Y[i] >= Y[i + 1] and Y[i] == np.max(Y[i - radius: i + radius + 1])):
                    ids.append(i)

            for i in range(1, len(ids)): # Unify peaks on the same finger
                i0 = ids[i - 1]
                i1 = ids[i]
                diff = np.max(Y[i0:i1+1]) - np.min(Y[i0:i1+1])
                if diff <= self.TIP_HEIGHT:
                    if Y[i1] < Y[i0]:
                        ids[i] = ids[i - 1]
                    ids[i - 1] = -1

            for i in range(len(ids)): # Judge finger height and width
                id = ids[i]
                if id != -1:
                    l = r = id
                    while (l - 1 >= 0 and Y[l - 1] >= Y[id] - self.TIP_HEIGHT and r-l<=self.THUMB_WIDTH):
                        l -= 1
                    while (r + 1 < point_num and Y[r + 1] >= Y[id] - self.TIP_HEIGHT and r-l<=self.THUMB_WIDTH):
                        r += 1
                    mid = (l + r) // 2

                    if X[mid] <= self.THUMB_R_EDGE or (np.min(finger[:,0]) == 0 and i == 0):
                        self.has_thumb = True # The finger on the left boundary is thumb
                        w_thres = self.THUMB_WIDTH
                    else:
                        w_thres = self.TIP_WIDTH
                    
                    if r - l <= w_thres and l != 0 and r != point_num-1:
                        fingertips.append([X[mid], Y[mid]])

        candidates = fingertips.copy() # There should be no pixel upper the fingertip
        fingertips = []
        for fingertip in candidates:
            x = fingertip[0]
            y = fingertip[1]
            ray = image[y:,x]
            if np.count_nonzero(ray) <= 10:
                fingertips.append(fingertip)
        fingertips = np.array(fingertips)

        if len(fingertips) > 5:
            order = np.argsort(fingertips[:,1])
            fingertips = fingertips[order[-5:]]

        return np.array(fingertips)

    def find_touch_line(self, image, threshold):
        A = cv2.resize(image, (self.M//2, self.N)) # Downsampling
            
        B = cv2.Laplacian(A, cv2.CV_8U, ksize=3) # Optional (cost=5ms)
        A = A - cv2.min(A,B)
        A = A.astype(np.int32)

        U = self.PALM_THRESHOLD # Up threshold
        D = self.N-200 # Down threshold. The touch line is between y=U and y=D
        A[:U,:] = 1e8
        A[D:,:] = 1e8

        for i in range(0,self.M//2-1):
            A[U:D,i+1] += cv2.min(cv2.min(A[U:D,i],A[U-1:D-1,i]),A[U+1:D+1,i])[:,0]

        i = self.M//2-1
        j = self.N-1-np.argmin(A[::-1,i])

        self.pressure = A[j,i]

        touch_line = np.zeros(self.M)
        while (i > 0):
            i -= 1
            if A[j+1,i] <= A[j,i] and A[j+1,i]<=A[j-1,i]: #j+1优先（描绘桌面边界）
                j += 1
            elif A[j-1,i] < A[j,i] and A[j-1,i]<A[j+1,i]:
                j -= 1
            touch_line[i*2] = j
            touch_line[i*2+1] = j
            image[j:,i*2] = 0
            image[j:,i*2+1] = 0

        return touch_line

    def assign_fingertips(self, fingertips):
        if len(fingertips) == 0:
            self.init_fingertips()
            return False
            
        order = np.argsort(fingertips[:,0])
        fingertips = fingertips[order]

        # Mapping fingers
        if len(fingertips) == 5:
            self.fingertips = fingertips.copy()
        elif len(fingertips) == 4 and self.has_thumb == False: # No thumb
            self.fingertips[0] = [-1, -1]
            self.fingertips[1:] = fingertips.copy()
        elif len(fingertips) == 4 and fingertips[-1][0] <= self.cx + 100: # No pinkie
            self.fingertips[:-1] = fingertips.copy()
            self.fingertips[-1] = [-1, -1]
        elif len(fingertips) == 3 and self.has_thumb == False and fingertips[-1][0] <= self.cx + 100: # No thumb and pinkie
            self.fingertips[0] = [-1, -1]
            self.fingertips[1:-1] = fingertips.copy()
            self.fingertips[-1] = [-1, -1]
        else: # Track the fingers: Greedy-based best match algorithm, max movement = 80 pixel/frame
            fingertip_num = len(fingertips)
            matrix = -np.ones((fingertip_num, 5))
            for i in range(fingertip_num):
                for j in range(5):
                    if self.fingertips[j][0] != -1:
                        pos = self.fingertips[j]
                        dist2 = (fingertips[i][0] - pos[0]) ** 2 + (fingertips[i][1] - pos[1]) ** 2
                        if dist2 <= self.MOVEMENT_THRESHOLD ** 2:
                            matrix[i][j] = dist2
            self.fingertips = [[-1, -1]] * 5
            while True:
                valid_index = np.where(matrix != -1)
                if len(valid_index[0]) == 0:
                    break
                id = np.argmin(matrix[valid_index])
                i = valid_index[0][id]
                j = valid_index[1][id]
                self.fingertips[j] = fingertips[i].copy()
                fingertips[i] = [-1,-1]
                matrix[i,:] = -1
                matrix[:,j] = -1

    def calc_touch(self, is_tapping, touch_line):
        last_is_touch = self.is_touch.copy()
        self.is_touch = [False for i in range(5)]
        self.is_touch_down = [False for i in range(5)]
        self.is_touch_up = [False for i in range(5)]
        
        if self.fingertips[0][0] != -1: # The thumb
            x, y = self.fingertips[0][0], self.fingertips[0][1]
            self.is_touch[0] = (y >= self.cy + self.SECOND_ROW_DELTA)
        if is_tapping:
            for i in range(1, 5): # Not the thumb
                if self.fingertips[i][0] != -1:
                    [x, y] = self.fingertips[i]
                    if y+3 >= touch_line[x]:
                        y=int(touch_line[x])
                        gray_line = cv2.cvtColor(self.image[y-10:y+10,x:x+1], cv2.COLOR_RGB2GRAY)
                        endpoint_brightness = float(np.min(gray_line)) / np.max(gray_line)
                        if endpoint_brightness >= self.MIN_ENDPOINT_BRIGHTNESS:
                            self.is_touch[i] = True
        
        for i in range(5):
            if self.fingertips[i][0] != -1:
                self.is_touch_down[i] = self.is_touch[i] and (not last_is_touch[i])
                self.is_touch_up[i] = (not self.is_touch[i]) and last_is_touch[i]
    
    def calc_endpoints(self):
        for i in range(5):
            if self.is_touch[i]:
                [X, Y] = self.fingertips[i]
                offset = (self.cy - Y) / (self.camera_H * self.fy / 80.0 + 1) # (80.0 = 8 * 10) Offset is -8 pixels when z = 10 cm
                Y += offset
                z = (self.camera_H * self.fy) / (Y - self.cy)
                x = (X - self.cx) / self.fx * z
                self.endpoints[i] = [x, z]
            else:
                self.endpoints[i] = [-1, -1]

    def run(self, image):
        self.image = image
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self.N, self.M = gray.shape
        brightness_threshold = self.find_brightness_threshold(gray)
        _, gray = cv2.threshold(gray, brightness_threshold, 255, type=cv2.THRESH_TOZERO)

        table_contour = self.find_table(gray, brightness_threshold)
        
        is_tapping = False
        touch_line = None
        if len(table_contour) != 0:
            if np.min(table_contour[:, :, 1]) <= self.PALM_THRESHOLD: # Table connect with fingers
                touch_line = self.find_touch_line(gray, brightness_threshold) # Also remove the table
                is_tapping = self.pressure >= self.MIN_PRESSURE
            else:
                cv2.drawContours(gray, [table_contour], -1, 0, cv2.FILLED)

        kernel = np.ones((3,3),np.uint8) # Remove small patches
        gray = cv2.erode(gray, kernel, iterations=4)
        gray = cv2.dilate(gray, kernel, iterations=4)

        fingertips = self.find_fingertips(gray, brightness_threshold)
        self.assign_fingertips(fingertips)
        self.calc_touch(is_tapping, touch_line)
        self.calc_endpoints()
        self.palm_line = self.find_palm_line(gray)

        # Save intermediate result
        self.illustration = gray
        if is_tapping:
            self.is_tapping = True
            self.touch_line = touch_line
        else:
            self.is_tapping = False

    def output(self, title=''):
        result = cv2.cvtColor(self.illustration, cv2.COLOR_GRAY2RGB)

        for i in range(len(self.fingertips)):
            fingertip = self.fingertips[i]
            if fingertip[0] == -1:
                continue
            is_touch = self.is_touch[i]
            if is_touch:
                color = (0,0,255)
            else:
                color = (0,255,0)
            cv2.circle(result, tuple(fingertip), 10, color, cv2.FILLED)
            cv2.putText(result, self.finger_names[i][0], tuple([fingertip[0]+15, fingertip[1]]), 0, 1.2, color, 3)
        if self.is_tapping:
            for i in range(self.M):
                cv2.circle(result, tuple([i,int(self.touch_line[i])]), 2, (255,0,0), cv2.FILLED)

        cv2.putText(result, title, tuple([0, 100]), 0, 1.2, (0,0,255), 3)
        result = cv2.resize(result, (320, 240))
        cv2.line(result, (int(self.cx)//4, 0), (int(self.cx)//4, self.N//4 - 1), (64,64,64), 1)
        cv2.line(result, (0, (int(self.cy)+self.SECOND_ROW_DELTA)//4), (self.M//4 - 1, (int(self.cy)+self.SECOND_ROW_DELTA)//4), (64,64,64), 1)
        cv2.line(result, (0, self.palm_line//4), (self.M//4 - 1, self.palm_line//4), (0,0,128), 1)
        cv2.line(result, (0, self.palm_height//4), (self.M//4 - 1, self.palm_height//4), (0,0,64), 1)
        #image = cv2.resize(self.image, (640, 380))
        #output = np.hstack([image, result])

        return result

