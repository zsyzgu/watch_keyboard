import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import io
import math
import pickle
from scipy.stats import kurtosis

class FingerTracker: # 1280 * 960
    CONTOURS_MIN_AREA = 6000 # the minimun area of contours of a finger
    PALM_THRESHOLD = 400 # the upper rows belong to the palm
    FINGER_GAP = 80 # The minimum distance between two finger in pixels on the contour
    FIXED_BRIGHTNESS = True

    def __init__(self, camera_id):
        self.init_fingertips()
        self.BRIGHTNESS_THRESHOLD = 100 # Default brightness threshold
        self.init_camera_para(camera_id)
    
    def init_fingertips(self):
        self.TIP_HEIGHT = 12 # Deal with two recognized points on a fingertip
        self.TIP_WIDTH = 180 # Max width of a tip (excluding thumb)
        self.THUMB_WIDTH = 300 # Max width of a thumb
        self.SECOND_ROW_DELTA = 50 # The second row of the keyboard is x pixels lower than camera_cy
        self.MOVEMENT_THRESHOLD = 80 # The maximun moving pixels of a fintertip in one frame
        self.MIN_ENDPOINT_BRIGHTNESS = 0.7 # The endpoint should be bright enough when contacting
        self.finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinkie']
        self.fingertips = [[-1, -1]] * 5
        self.endpoints = [[-1,-1]] * 5 # endpoint on the desk calned by touchpoint
        self.is_touch = [False for i in range(5)]
        self.is_touch_down = [False for i in range(5)]
        self.is_touch_up = [False for i in range(5)]
        self.curr_middle_finger = [-1, -1] # The tip of the middle finger in the latest available frame
        self.palm_line = 0 # The root of the middle finger (in y axis)
        self.palm_line_dx = 0 # (dx,dy) = the vector between the groove of 2nd/3rd fingers and the groove of 3rd/4th fingers
        self.palm_line_dy = 0

    def init_camera_para(self, camera_id):
        assert(camera_id == 1 or camera_id == 2)
        self.camera_id = camera_id
        [self.map1, self.map2, self.roi, self.camera_mtx] = pickle.load(open(str(camera_id) + '.calib', 'rb'))
        self.camera_H = 0.75
        self.fx = self.camera_mtx[0][0]
        self.fy = self.camera_mtx[1][1]
        self.cx = self.camera_mtx[0][2]
        self.cy = self.camera_mtx[1][2]

    def find_contours(self, frame, threshold, hier):
        _, binary = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, hier, cv2.CHAIN_APPROX_NONE)
        return contours

    def find_table(self, contours): # The table is the max contour connect with the bottom
        bottoms = []
        for contour in contours:
            if (self.N-1) in contour[:, :, 1]:
                bottoms.append(contour)
        if len(bottoms) == 0:
            return []
        table_contour = max(bottoms, key=cv2.contourArea)
        return table_contour

    def find_fingers(self, contours): # Find chunks containing fingers
        fingers = []
        for contour in contours:
            if cv2.contourArea(contour) >= self.CONTOURS_MIN_AREA: # The finger must have enough area
                points = contour[:,0]
                fingers.append(np.array(points))

        return fingers

    def find_brightness_threshold(self):
        if self.FIXED_BRIGHTNESS:
            if self.camera_id == 1:
                return 53
            if self.camera_id == 2:
                return 53

    def find_palm_line(self): # palm_line = the root of the middel finger
        if self.fingertips[2][0] == -1:
            return self.palm_line
            
        [X, Y, mid] = self.finger_map[self.fingertips[2][0]]
        N = len(X)
        
        radius = self.FINGER_GAP // 2
        l = max(0, mid - radius)
        r = min(N-1, mid + radius)

        cnt = 0
        while (l - 1 >= 0 and cnt < 5):
            if Y[l-1] < Y[l]:
                cnt = 0
            else:
                cnt += 1
            l -= 1
        l += cnt

        cnt = 0
        while (r + 1 < N and cnt < 5):
            if Y[r+1] < Y[r]:
                cnt = 0
            else:
                cnt += 1
            r += 1
        r -= cnt
        
        xl, xr = X[l], X[r]
        yl, yr = Y[l], Y[r]
        x = (xl + xr) // 2
        y = (yl + yr) // 2

        if self.camera_id == 2: # Right hand
            missing = yr-yl > 60 or yl-yr > 10
        else:
            missing = yl-yr > 60 or yr-yl > 10
        if missing:
            if yl < yr:
                x = xl + self.palm_line_dx // 2
                y = yl + self.palm_line_dy // 2
            else:
                x = xr - self.palm_line_dx // 2
                y = yr - self.palm_line_dy // 2
        else:
            self.palm_line_dx = xr - xl
            self.palm_line_dy = yr - yl

        return y

    def find_fingertips(self, contours):
        fingers = self.find_fingers(contours)
        fingertips = []
        self.has_thumb = False
        
        if len(fingers) > 0:
            small_chunk = np.min([len(finger) for finger in fingers]) # The thumb can only on the smallest chunk
        self.finger_map = {} # Record the edge around fingertips, KEY = x of a fingertip

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
            while -1 in ids:
                ids.remove(-1)

            if self.camera_id == 1: # Left hand
                maybe_thumb = (len(finger) == small_chunk) and (np.max(finger[:,0]) == self.M-1)
                ids.reverse()
            else: # Right hand
                maybe_thumb = (len(finger) == small_chunk) and (np.min(finger[:,0]) == 0)

            for i in range(len(ids)): # Judge finger height and width
                id = ids[i]
                l = r = id
                while (l - 1 >= 0 and Y[l - 1] >= Y[id] - self.TIP_HEIGHT and r-l<=self.THUMB_WIDTH):
                    l -= 1
                while (r + 1 < point_num and Y[r + 1] >= Y[id] - self.TIP_HEIGHT and r-l<=self.THUMB_WIDTH):
                    r += 1
                mid = (l + r) // 2

                #self.THUMB_R_EDGE = 300 # the left cols just belong to the thumb
                #if maybe_thumb or (self.camera_id == 2 and X[mid] <= self.THUMB_R_EDGE) or (self.camera_id == 1 and self.M-1-X[mid] <= self.THUMB_R_EDGE):
                if maybe_thumb:
                    w_thres = self.THUMB_WIDTH
                else:
                    w_thres = self.TIP_WIDTH

                if r - l <= w_thres and (maybe_thumb or (X[l] != 0 and X[r] != self.M-1)):
                    [x,y] = [X[mid],Y[mid]]
                    if maybe_thumb:
                        self.has_thumb = True
                    maybe_thumb = False
                    fingertips.append([x,y])
                    self.finger_map[x] = [X,Y,mid]

        fingertips = np.array(fingertips)
        if fingertips.shape[0] > 5:
            order = np.argsort(fingertips[:,1])
            fingertips = fingertips[order[-5:],:]

        return fingertips

    def find_touch_line(self, image, threshold):
        table_contour = []
        A = cv2.resize(image, (self.M//2, self.N)) # Downsampling
        B = cv2.Laplacian(A, cv2.CV_8U, ksize=3) # Optional (cost=5ms)
        A = A - cv2.min(A,B)
        A = A.astype(np.int32)

        U = self.PALM_THRESHOLD # Up threshold
        D = self.N-200 # Down threshold. The touch line is between y=U and y=D
        A[:U,:] = 1e8
        A[D:,:] = 1e8

        for i in range(self.M//2-1):
            A[U:D,i+1] += cv2.min(cv2.min(A[U:D,i],A[U-1:D-1,i]),A[U+1:D+1,i])[:,0]

        i = self.M//2-1
        j = self.N-1-np.argmin(A[::-1,i])

        table_contour.append([[self.M,j]])

        touch_line = np.zeros(self.M)
        while (i > 0):
            i -= 1

            if A[j+1,i] <= A[j,i] and A[j+1,i]<=A[j-1,i]: #j+1优先（描绘桌面边界）
                j += 1
            elif A[j-1,i] < A[j,i]:
                j -= 1

            touch_line[i*2:i*2+2] = j
            table_contour.append([[i*2,j]])
        
        table_contour.append([[0,j]])
        table_contour.append([[0,self.N]])
        table_contour.append([[self.M,self.N]])

        return touch_line, np.array(table_contour)

    def assign_fingertips(self, fingertips):
        if len(fingertips) == 0:
            self.init_fingertips()
            return False
            
        order = np.argsort(fingertips[:,0])
        if self.camera_id == 1: # Left hand
            order = order[::-1]
        fingertips = fingertips[order]
        if self.camera_id == 1: # Left hand
            has_pinkie = (fingertips[-1][0] <= self.cx - 200)
        else: # Right hand
            has_pinkie = (fingertips[-1][0] >= self.cx + 200)

        # Mapping fingers
        if len(fingertips) == 5:
            self.fingertips = fingertips.copy()
        elif len(fingertips) == 4 and has_pinkie == False: # No pinkie
            self.fingertips[:-1] = fingertips.copy()
            self.fingertips[-1] = [-1, -1]
        elif len(fingertips) == 4 and self.has_thumb == False: # No thumb
            self.fingertips[0] = [-1, -1]
            self.fingertips[1:] = fingertips.copy()
        elif len(fingertips) == 3 and self.has_thumb == False and has_pinkie == False: # No thumb and pinkie
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

    def calc_touch(self, has_touch_line, touch_line):
        image = self.image
        last_is_touch = self.is_touch.copy()
        self.is_touch = [False for i in range(5)]
        self.is_touch_down = [False for i in range(5)]
        self.is_touch_up = [False for i in range(5)]
        
        if self.fingertips[0][0] != -1: # The thumb
            x, y = self.fingertips[0][0], self.fingertips[0][1]
            self.is_touch[0] = (y >= self.cy + self.SECOND_ROW_DELTA)
        if has_touch_line:
            for i in range(1, 5): # Not the thumb
                if self.fingertips[i][0] != -1:
                    [x, y] = self.fingertips[i]
                    if y+5 >= touch_line[x]:
                        y=int(touch_line[x])
                        gray_line = cv2.cvtColor(image[y-10:y+10,x:x+1], cv2.COLOR_RGB2GRAY)
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
                X, Y = self.map1[Y, X] # Undistortion
                if self.camera_id == 1:
                    offset = (self.cy - Y) / (self.camera_H * self.fy / 90.0 + 1) # (90.0 = 9 * 10) Offset is -9 pixels when z = 10 cm
                else:
                    offset = (self.cy - Y) / (self.camera_H * self.fy / 170.0 + 1) # Offset is -17 pixels when z = 10 cm
                Y += offset
                z = (self.camera_H * self.fy) / (Y - self.cy)
                x = (X - self.cx) / self.fx * z
                self.endpoints[i] = [x, z]
            else:
                self.endpoints[i] = [-1, -1]

    def erode_fingers(self, frame, brightness_threshold):
        kernel = np.uint8(np. ones((1, 5)))

        sob1 = cv2.Sobel(frame, -1, 1, 0, ksize=-1)
        r1 = cv2.subtract(frame, sob1)
        _, r1 = cv2.threshold(r1, brightness_threshold, 255, type=cv2.THRESH_TOZERO)
        r1 = cv2.erode(r1, kernel, iterations=3)

        sob2 = cv2.flip(cv2.Sobel(cv2.flip(frame, 1), -1, 1, 0, ksize=-1), 1)
        r2 = cv2.subtract(frame, sob2)
        _, r2 = cv2.threshold(r2, brightness_threshold, 255, type=cv2.THRESH_TOZERO)
        r2 = cv2.erode(r2, kernel, iterations=3)

        r = cv2.max(r1,r2)
        return r

    def run(self, image):
        t=time.clock()
        self.N, self.M = image.shape[:2]
        self.image = image
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        brightness_threshold = self.find_brightness_threshold()
        _, gray = cv2.threshold(gray, brightness_threshold, 255, type=cv2.THRESH_TOZERO)

        kernel = np.ones((3,3),np.uint8) # Remove small patches
        gray = cv2.erode(gray, kernel, iterations=4)
        gray = cv2.dilate(gray, kernel, iterations=4)

        contours = self.find_contours(gray, brightness_threshold, cv2.RETR_EXTERNAL)
        table_contour = self.find_table(contours)
        
        has_touch_line = False
        touch_line = None
        if len(table_contour) != 0:
            if np.min(table_contour[:, :, 1]) <= self.PALM_THRESHOLD: # Table connect with fingers
                touch_line, table_contour = self.find_touch_line(gray, brightness_threshold) # Also remove the table
                has_touch_line = True
            cv2.drawContours(gray, [table_contour], -1, 0, cv2.FILLED) # There is a table, but not connect with fingers
        
        gray_erode_fingers = self.erode_fingers(gray, brightness_threshold)
        contours = self.find_contours(gray_erode_fingers, brightness_threshold, cv2.RETR_EXTERNAL)
        self.fingers_contours = contours
        
        fingertips = self.find_fingertips(contours)
        self.assign_fingertips(fingertips)
        self.calc_touch(has_touch_line, touch_line)
        self.calc_endpoints()
        self.palm_line = self.find_palm_line()

        # Save intermediate result
        self.illustration = gray
        if has_touch_line:
            self.has_touch_line = True
            self.touch_line = touch_line
        else:
            self.has_touch_line = False
            
        print(time.clock()-t)

    def output(self, title=''):
        result = self.illustration
        #result = cv2.drawContours(result, self.fingers_contours, -1, 192, 3)
        
        for i in range(len(self.fingertips)):
            fingertip = self.fingertips[i]
            if fingertip[0] == -1:
                continue
            is_touch = self.is_touch[i]
            if is_touch:
                size = 20
            else:
                size = 10
            cv2.circle(result, tuple(fingertip), size, 255, cv2.FILLED)
            cv2.putText(result, self.finger_names[i][0], tuple([fingertip[0]+15, fingertip[1]]), 0, 1.2, 255, 3)

        #if self.has_touch_line: # Draw touch line
        #    for i in range(self.M):
        #        cv2.circle(result, tuple([i,int(self.touch_line[i])]), 2, (255,0,0), cv2.FILLED)

        cv2.putText(result, title, tuple([0, 100]), 0, 1.2, (0,0,255), 3)
        result = cv2.resize(result, (320, 240))
        cv2.line(result, (int(self.cx)//4, 0), (int(self.cx)//4, self.N//4 - 1), (64,64,64), 1)
        cv2.line(result, (0, int(self.cy)//4), (self.M//4 - 1, int(self.cy)//4), (64,64,64), 1)
        cv2.line(result, (0, self.palm_line//4), (self.M//4 - 1, self.palm_line//4), (0,0,128), 1)
        
        return result#cv2.resize(self.image, (320, 240))

