import numpy as np
import cv2
from finger_tracker import FingerTracker
import time
import pickle
import img2video
import os
import matplotlib.pyplot as plt
import pickle
import sys
import pygame
from matplotlib.patches import Ellipse
from astroML.stats import fit_bivariate_normal
from astroML.stats.random import bivariate_normal

class Visual:
    GRID = 50

    def __init__(self):
        self.init_letter_positions()
        self.init_display()

    def init_letter_positions(self):
        QWERTY = ['QWERTYUIOP', 'ASDFGHJKL', 'ZXCVBNM']
        FINGER_PINKIE = 'QAZ|P'
        FINGER_RING = 'WSX|OL'
        FINGER_MIDDLE = 'EDC|IK'
        FINGER_INDEX_L = 'RFV|TGB'
        FINGER_INDEX_R = 'YHN|UJM'
        self.letter_positions = [[-1, -1] for i in range(26)]
        self.letter_colors = [(0,0,0) for i in range(26)]
        for r in range(3):
            line = QWERTY[r]
            for c in range(len(line)):
                ch = line[c]
                index = ord(ch) - ord('A')
                self.letter_positions[index] = [c, r]
                color = (0,0,0)
                I = 64
                if ch in FINGER_PINKIE:
                    color = (0,I,0)
                elif ch in FINGER_RING:
                    color = (I,0,I)
                elif ch in FINGER_MIDDLE:
                    color = (I,I,0)
                elif ch in FINGER_INDEX_L:
                    color = (0,I,I)
                elif ch in FINGER_INDEX_R:
                    color = (0,0,I)
                self.letter_colors[index] = color
    
    def init_display(self):
        pygame.init()
        self.screen = pygame.display.set_mode((10 * self.GRID + 1, 4 * self.GRID + 1))
        pygame.display.set_caption('Register')
        self.highlight_R = -1
        self.highlight_C = -1
    
    def draw(self):
        GRID = self.GRID
        image = np.zeros((4 * GRID + 1, 10 * GRID + 1, 3), np.uint8)

        cv2.rectangle(image, (0, 0), (10 * GRID, GRID - 1), (0, 0, 0), -1)

        # Draw the keyboard layout
        for r in range(3):
            for c in range(10):
                ch = '_'
                bg_color = (64, 64, 64)
                for i in range(26):
                    pos = self.letter_positions[i]
                    if c == pos[0] and r == pos[1]:
                        ch = chr(i + ord('A'))
                        bg_color = self.letter_colors[i]
                cv2.rectangle(image, (int(c * GRID), int((r + 1) * GRID)), (int((c + 1) * GRID), int((r + 2) * GRID)), bg_color, -1)
                cv2.rectangle(image, (int(c * GRID), int((r + 1) * GRID)), (int((c + 1) * GRID), int((r + 2) * GRID)), (255, 255, 255), 1)
                cv2.putText(image, ch, (int(c * GRID) + 15, int((r + 2) * GRID) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                if r == self.highlight_R and c == self.highlight_C:
                    image[int((r + 1) * GRID): int((r + 2) * GRID), int(c * GRID): int((c + 1) * GRID)] *= 2

        pg_img = pygame.surfarray.make_surface(cv2.transpose(image))
        self.screen.blit(pg_img, (0,0))
        pygame.display.flip()

    def update_RC(self, camera_id, r, c):
        assert(camera_id == 1 or camera_id == 2)
        if camera_id == 1:
            self.highlight_R = r
            self.highlight_C = 4 - c
        else:
            self.highlight_R = r
            self.highlight_C = 5 + c

    def get_keyboard_events(self):
        keys = []
        for event in pygame.event.get():
            if event.type == pygame.KEYUP:
                keys.append(event.key)
        return keys

def record(camera_id):
    camera = cv2.VideoCapture(camera_id - 1)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

    tracker = FingerTracker(camera_id)
    visaul = Visual()

    index = 0
    frames = []
    while True:
        succ, frame = camera.read()
        if succ == False:
            break
        tracker.run(frame)
        output = tracker.output(str(index))
        cv2.imshow('illustration', output)
        cv2.waitKey(1)

        keys = visaul.get_keyboard_events()
        if pygame.K_q in keys:
            break

        iteration = int(index // 15)
        row = int((index % 15) // 5)
        col = int(index % 5)
        visaul.update_RC(camera_id, row, col)

        finger = max(1, col)
        correct = tracker.is_touch_down[finger]
        if correct:
            x = tracker.endpoints[finger][0]
            y = tracker.palm_line
            frames.append([frame, x, y])
            index += 1
            if index == 150:
                break

        visaul.draw()
    
    return frames

def calc(camera_id, frames):
    pc = np.zeros((3, 5, 5))
    fig, ax = plt.subplots()

    for r in range(3):
        print(r)
        for c in range(5):
            X = []
            Y = []
            for index in range(150):
                iteration = int(index // 15)
                row = int((index % 15) // 5)
                col = int(index % 5)
                if row == r and col == c:
                    [frame, x, y] = frames[index]
                    X.append(x)
                    Y.append(y)
                            
            n_std = 3
            xc, x_std = np.mean(X), np.std(X)
            yc, y_std = np.mean(Y), np.std(Y)
            pack = zip(X.copy(), Y.copy())
            X = []
            Y = []
            for x, y in pack:
                if abs(x-xc) <= n_std * x_std and abs(y-yc) <= n_std * y_std:
                    X.append(x)
                    Y.append(y)

            X = np.array(X)
            Y = np.array(Y)
            assert(len(X) > 0)
            print('R = %d, C = %d, NUM = %d' % (r,c,len(X)))
            (center, a, b, theta) = fit_bivariate_normal(X, Y, robust=True)
            xc, yc = center
            ell = Ellipse(center, a * n_std, b * n_std, (theta * 180. / np.pi), ec='k', fc='none', color='red')            

            index = r * 5 + c + 1
            plt.scatter(X, Y, color=('C'+str(index)), s = 5)
            plt.scatter(xc, yc, color='red', s = 10)
            ax.add_patch(ell)

            pc[r,c] = [xc, yc, a, b, theta]

    plt.show()    
    pickle.dump(pc, open(str(camera_id) + '.regist', 'wb'))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('[Usage] python register.py camera_id')
        exit()
    camera_id = int(sys.argv[1])
    assert(1 <= camera_id and camera_id <= 2)
    frames = record(camera_id)
    #frames = pickle.load(open('debug.pickle','rb'))
    pickle.dump(frames,open('debug.pickle','wb'))
    calc(camera_id, frames)
