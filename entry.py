import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import io
import os
import pygame
from keyboard import Keyboard
from finger_tracker import FingerTracker
import pickle
import sys

class Exp:
    def __init__(self):
        pygame.init()
        self.keyboard = Keyboard(VISABLE_FEEDBACK=Keyboard.VISABLE_ALWAYS, WORD_CORRECTION=Keyboard.CORRECT_LETTER)
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
    
    def acquire_frame(self):
        succ_L, frame_L = self.camera_L.read()
        succ_R, frame_R = self.camera_R.read()
        if succ_L and succ_R:
            return True, frame_L, frame_R
        else:
            return False, None, None
    
    def get_keyboard_events(self):
        keys = []
        for event in pygame.event.get():
            if event.type == pygame.KEYUP:
                keys.append(event.key)
        return keys

def calc_letter(keyboard, input_data):
    [side, index, highlight_row, highlight_col] = input_data[:4]
    row = int(round(max(0,min(2,highlight_row - 1))))
    col = int(round(max(0,min(1,highlight_col - 1))))
    if side == 'L':
        if index == 1:
            col = 3 + col
        else:
            col = 3 - (index - 1)
    if side == 'R':
        if index == 1:
            col = 6 - col
        else:
            col = 6 + (index - 1)
    
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

class Exp1(Exp):
    def __init__(self):
        super().__init__()

        self.save_folder = save_folder = 'data/' + sys.argv[1] + '/'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        else:
            print('\033[1;31;40m[Warning]\033[0m Folder exists')
        if not os.path.exists(save_folder + 'L/'):
            os.makedirs(save_folder + 'L/')
        if not os.path.exists(save_folder + 'R/'):
            os.makedirs(save_folder + 'R/')
        self.frame_id = 0

    def update_hightlight(self):
        self.keyboard.L_row = self.tracker_L.highlight_row
        self.keyboard.L_col = self.tracker_L.highlight_col
        self.keyboard.R_row = self.tracker_R.highlight_row
        self.keyboard.R_col = self.tracker_R.highlight_col

    def pack_input_data(self):
        # [side, which_finger, highlight_row, highlight_col, timestamp, palm_line, endpoint_x, endpoint_y, image_L, image_R]
        for i in range(5):
            tracker = None
            if self.tracker_L.is_touch_down[i]:
                side = 'L'
                tracker = self.tracker_L
            if self.tracker_R.is_touch_down[i]:
                side = 'R'
                tracker = self.tracker_R
            if tracker != None:
                return [side, i, tracker.highlight_row, tracker.highlight_col, time.clock(), tracker.palm_line, tracker.endpoints[i][0], tracker.endpoints[i][1], self.tracker_L.image.copy(), self.tracker_R.image.copy()]
        return []

    def save_data(self):
        data = self.keyboard.inputted_data
        for i in range(len(data)):
            image_L = data[i][-2]
            image_R = data[i][-1]
            data[i] = data[i][:-2] + [self.frame_id]
            cv2.imwrite(self.save_folder + 'L/' + str(self.frame_id) + '.jpg', image_L)
            cv2.imwrite(self.save_folder + 'R/' + str(self.frame_id) + '.jpg', image_R)
            self.frame_id += 1
        save_file = open(self.save_folder + str(self.keyboard.curr_task_id) + '.pickle', 'wb')
        pickle.dump([self.keyboard.task, self.keyboard.inputted_text, data], save_file)

    def run(self):
        self.keyboard.draw()

        is_running = True
        while is_running:
            succ, image_L, image_R = self.acquire_frame()
            if not succ:
                break
            self.tracker_L.run(image_L)
            self.tracker_R.run(image_R)

            if True:
                output_L = self.tracker_L.output()
                output_R = self.tracker_R.output()
                output = np.hstack([output_L, output_R])
                cv2.imshow('illustration', output)
                cv2.waitKey(1)
            
            self.update_hightlight()
            input_data = self.pack_input_data()
            keys = self.get_keyboard_events()

            if (len(input_data) > 0 and input_data[:2] == ['R', 0]) or pygame.K_SPACE in keys: # Enter a space (R - Thumb)
                self.keyboard.enter_a_space(input_data)
                end_time = time.clock()
            
            if (len(input_data) > 0 and input_data[1] >= 1): # Enter a letter (L/R - Index to Pinkie)
                letter = calc_letter(self.keyboard, input_data)
                if letter != '-':
                    if self.keyboard.inputted_text == '':
                        start_time = time.clock()
                    else:
                        end_time = time.clock()
                    self.keyboard.enter_a_letter(input_data, letter)
                else:
                    self.keyboard.delete_a_letter()
            
            if (len(input_data) > 0 and input_data[:2] == ['L', 0]) or pygame.K_n in keys: # Next phrase (L - Thumb)
                if len(self.keyboard.inputted_text) == len(self.keyboard.task):
                    wpm = (len(self.keyboard.inputted_text)-1)/((end_time - start_time)/60.0)/5.0
                    print('WPM = ', wpm)
                    self.save_data()
                    is_running = self.keyboard.next_phrase()
            
            if pygame.K_r in keys: # Redo phrase
                self.keyboard.redo_phrase()
            if pygame.K_q in keys: # Quit
                is_running = False
                
            self.keyboard.draw()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('[Usage] python entry.py save_folder')
        exit()
    exp = Exp1()
    exp.run()
