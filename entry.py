import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import io
import os
import pygame
pygame.init()
from keyboard import Keyboard
from finger_tracker import FingerTracker
import pickle

class Exp:
    def __init__(self):
        self.keyboard = Keyboard()
        self.tracker_L = FingerTracker(1)
        self.tracker_R = FingerTracker(2)
        self.init_camera()

    def init_camera(self):
        self.camera_L = cv2.VideoCapture(0)
        self.camera_L.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera_L.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
        self.camera_R = cv2.VideoCapture(1)
        self.camera_R.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera_R.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
        [self.map1_L, self.map2_L, _, _] = pickle.load(open('1.calib', 'rb'))
        [self.map1_R, self.map2_R, _, _] = pickle.load(open('2.calib', 'rb'))
    
    def acquire_frame(self):
        succ_L, frame_L = self.camera_L.read()
        succ_R, frame_R = self.camera_R.read()
        if succ_L and succ_R:
            frame_L = cv2.remap(frame_L, self.map1_L, self.map2_L,cv2.INTER_LINEAR,borderValue=cv2.BORDER_CONSTANT)
            frame_R = cv2.remap(frame_R, self.map1_R, self.map2_R,cv2.INTER_LINEAR,borderValue=cv2.BORDER_CONSTANT)
            frame_L = cv2.flip(frame_L, 1)
            return True, frame_L, frame_R
        else:
            return False, None, None
    
    def get_keyboard_events(self):
        keys = []
        for event in pygame.event.get():
            if event.type == pygame.KEYUP:
                keys.append(event.key)
        return keys

def calc_letter(side, index, keyboard):
    if side == 'L':
        row = max(0,min(2,int(round(keyboard.hl_L_row - 1))))
        if index == 1:
            if keyboard.hl_L_col >= 1.5:
                col = 4
            else:
                col = 3
        else:
            col = 4 - index
    if side == 'R':
        row = max(0,min(2,int(round(keyboard.hl_R_row - 1))))
        if index == 1:
            if keyboard.hl_R_col >= 1.5:
                col = 5
            else:
                col = 6
        else:
            col = 5 + index
    
    for i in range(26):
        [c, r] = keyboard.letter_positions[i]
        if row == r and col == c:
            return chr(ord('a') + i)
    
    if row == 2 and col == 7:
        return 'k'
    if row == 2 and col == 8:
        return 'l'
    if (row == 1 and col == 9) or (row == 2 and col == 9):
        return '-'

    return '#'

class Exp3(Exp):
    def __init__(self):
        super().__init__()

    def update_hightlight(self):
        hl_L_row = (self.tracker_L.cy - self.tracker_L.palm_line) * 0.01-0.5
        hl_R_row = (self.tracker_R.cy - self.tracker_R.palm_line) * 0.01-0.5
        hl_L_col = None
        hl_R_col = None
        if self.tracker_L.fingertips[1][0] != -1:
            hl_L_col = float(self.tracker_L.fingertips[1][0] - self.tracker_L.cx + 25) / (-50)
            # A Trick
            hl_L_col = hl_L_col + 1.5 - (hl_L_row * 0.71 + 0.26)
        if self.tracker_R.fingertips[1][0] != -1:
            hl_R_col = float(self.tracker_R.fingertips[1][0] - self.tracker_R.cx + 25) / (-50)
            # A Trick
            hl_R_col = hl_R_col + 1.5 - (hl_R_row * 0.71 + 0.26)
        self.keyboard.update_hightlight(hl_L_row, hl_L_col, hl_R_row, hl_R_col)
        self.keyboard.draw()

    def run(self):
        self.keyboard.draw()

        is_running = True
        is_illustration = True
        while is_running:
            succ, image_L, image_R = self.acquire_frame()
            if not succ:
                break
            self.tracker_L.run(image_L)
            self.tracker_R.run(image_R)

            if is_illustration:
                output_L = self.tracker_L.output()
                output_L = cv2.flip(output_L, 1)
                output_R = self.tracker_R.output()
                output = np.hstack([output_L, output_R])
                cv2.imshow('illustration', output)
                cv2.waitKey(1)
            
            self.update_hightlight()

            keys = self.get_keyboard_events()
            if self.tracker_R.is_touch_down[0] or pygame.K_SPACE in keys: # Entry a space
                self.keyboard.enter_a_space()
            if (True in self.tracker_L.is_touch_down[1 : 5]) or (True in self.tracker_R.is_touch_down[1 : 5]): # Entry a letter
                for i in range(1, 5):
                    if self.tracker_L.is_touch_down[i]:
                        side = 'L'
                        index = i
                        position = self.tracker_L.fingertips[i]
                        endpoint = self.tracker_L.endpoints[i]
                        palm_line = self.tracker_L.palm_line
                    if self.tracker_R.is_touch_down[i]:
                        side = 'R'
                        index = i
                        position = self.tracker_R.fingertips[i]
                        endpoint = self.tracker_R.endpoints[i]
                        palm_line = self.tracker_R.palm_line
                letter = calc_letter(side, index, self.keyboard)
                if letter != '-':
                    self.keyboard.enter_a_letter(letter)
                else:
                    self.keyboard.delete_a_letter()
            if pygame.K_i in keys: # open/close illustration
                is_illustration ^= True
                if is_illustration == False:
                    cv2.destroyAllWindows()
            if pygame.K_q in keys: # Quit
                is_running = False
            if pygame.K_n in keys or self.tracker_L.is_touch_down[0]: # Next phrase
                if len(self.keyboard.inputted_text) == len(self.keyboard.task_list[self.keyboard.curr_task_id]):
                    succ = self.keyboard.next_phrase()
                    if not succ:
                        is_running = False
            if pygame.K_r in keys: # Redo phrase
                self.keyboard.redo_phrase()

if __name__ == "__main__":
    exp = Exp3()
    exp.run()
