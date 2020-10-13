import cv2
import numpy as np
import time
import random
import pygame
import math
from PIL import Image, ImageFont, ImageDraw
from decoder import Decoder

class Keyboard:
    GRID = 50
    TASK_NUM = 20
    CORPUS_NUM = 20000

    VISABLE_NO = 0
    VISABLE_TOUCH = 1
    VISABLE_ALWAYS = 2
    CORRECT_NO = 0
    CORRECT_WORD = 1
    CORRECT_LETTER = 2
    
    def __init__(self, VISABLE_FEEDBACK = VISABLE_ALWAYS, WORD_CORRECTION = CORRECT_WORD):
        self.VISABLE_FEEDBACK = VISABLE_FEEDBACK
        self.WORD_CORRECTION = WORD_CORRECTION
        self.init_letter_info()
        self.init_task_list('phrases.txt')
        self.init_corpus()
        self.init_inputted_data()
        self.init_display()
    
    def init_letter_info(self):
        self.decoder = Decoder()
        
        FINGER_PINKIE = 'QAZ|P'
        FINGER_RING = 'WSX|OL'
        FINGER_MIDDLE = 'EDC|IK'
        FINGER_INDEX_L = 'RFV|TGB'
        FINGER_INDEX_R = 'YHN|UJM'
        self.letter_colors = [(0,0,0) for i in range(26)]
        for index in range(26):
            ch = chr(index + ord('A'))
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

    def init_task_list(self, path):
        self.task_list = []
        self.curr_task_id = 0

        lines = open(path).readlines()
        for line in lines:
            line = line.lower()
            self.task_list.append(line.strip('\n'))

        random.shuffle(self.task_list)
        self.task_list = self.task_list[:self.TASK_NUM]
        self.task = self.task_list[self.curr_task_id]

    def init_corpus(self):
        if self.WORD_CORRECTION:
            self.corpus = []
            lines = open('corpus.txt').readlines()
            for i in range(self.CORPUS_NUM):
                line = lines[i]
                tags = line.strip().split(' ')
                word = tags[0]
                pri = float(tags[1])
                self.corpus.append((word, pri))

    def init_inputted_data(self):
        self.inputted_text = ''
        self.inputted_data = []
        self.last_touch_time = -1

    def init_display(self):
        self.screen = pygame.display.set_mode((10 * self.GRID + 1, 4 * self.GRID + 1))
        pygame.display.set_caption('Qwerty Watch')
        self.L_row = None # Hightline line
        self.L_col = None
        self.R_row = None
        self.R_col = None

    def draw(self):
        GRID = self.GRID
        image = np.zeros((4 * GRID + 1, 10 * GRID + 1, 3), np.uint8)

        cv2.rectangle(image, (0, 0), (10 * GRID, GRID - 1), (0, 0, 0), -1)

        # Draw task and inputted text
        cv2.putText(image, self.task, (int(GRID * 0.5), int(GRID * 0.4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(image, self.inputted_text + '_', (int(GRID * 0.5), int(GRID * 0.8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Draw the keyboard layout
        for i in range(26):
            ch = chr(i + ord('A'))
            pos = self.decoder.positions[i]
            bg_color = self.letter_colors[i]
            cv2.rectangle(image, (int(pos[0] * GRID), int((pos[1] + 1) * GRID)), (int((pos[0] + 1) * GRID), int((pos[1] + 2) * GRID)), bg_color, -1)
            cv2.rectangle(image, (int(pos[0] * GRID), int((pos[1] + 1) * GRID)), (int((pos[0] + 1) * GRID), int((pos[1] + 2) * GRID)), (255, 255, 255), 1)
            cv2.putText(image, ch, (int(pos[0] * GRID) + 15, int((pos[1] + 2) * GRID) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Visable feedback
        if self.VISABLE_FEEDBACK == self.VISABLE_ALWAYS:
            if self.L_row != None:
                row = max(0.5, min(3.5, self.L_row))
                row_pixel = int((row - 0.5 + 1) * GRID)
                image[row_pixel-2:row_pixel+3,:5*GRID] *= 2
            if self.R_row != None:
                row = max(0.5, min(3.5, self.R_row))
                row_pixel = int((row - 0.5 + 1) * GRID)
                image[row_pixel-2:row_pixel+3,5*GRID:] *= 2
            if self.L_col != None:
                col = max(0.5, min(2.5, self.L_col))
                col_pixel = int((2.5 + col) * GRID)
                image[1*GRID:4*GRID,col_pixel-2:col_pixel+3] *= 2
            if self.R_col != None:
                col = max(0.5, min(2.5, self.R_col))
                col_pixel = int((7.5 - col) * GRID)
                image[1*GRID:4*GRID,col_pixel-2:col_pixel+3] *= 2
        elif self.VISABLE_FEEDBACK == self.VISABLE_TOUCH:
            DURATION = 0.5
            if time.clock() - self.last_touch_time < DURATION and len(self.inputted_data) > 0:
                [col, row] = self.get_position(self.inputted_data[-1])
                row_pixel = int((row - 0.5 + 2) * GRID)
                col_pixel = int((col - 0.5 + 1) * GRID)
                schedule = (time.clock() - self.last_touch_time) / DURATION
                image[row_pixel-5:row_pixel+6,col_pixel-5:col_pixel+6] = cv2.add(image[row_pixel-5:row_pixel+6,col_pixel-5:col_pixel+6], int(255 * (1 - schedule)))
        elif self.VISABLE_FEEDBACK == self.VISABLE_NO:
            pass

        pg_img = pygame.surfarray.make_surface(cv2.transpose(image))
        self.screen.blit(pg_img, (0,0))
        pygame.display.flip()
    
    def next_phrase(self):
        self.curr_task_id += 1
        print('Phase = %d' % (self.curr_task_id))
        self.inputted_text = ''
        self.inputted_data = []
        if self.curr_task_id >= len(self.task_list):
            self.curr_task_id = 0
            return False
        self.task = self.task_list[self.curr_task_id]
        return True

    def redo_phrase(self):
        self.inputted_text = ''
        self.inputted_data = []

    def enter_a_letter(self, input_data, input_letter):
        i = len(self.inputted_text)
        letter = ''
        if i < len(self.task):
            if self.WORD_CORRECTION == self.CORRECT_LETTER:
                if self.task[i] == ' ': # can not enter space by inputting letter, when CORRECT_LETTER
                    return ''
                letter = self.task[i]
            else:
                letter = input_letter
            self.inputted_text += letter
            self.inputted_data.append(input_data)
            self.last_touch_time = time.clock() 
        return letter

    def enter_a_space(self, input_data):
        i = len(self.inputted_text)
        if i == 0 or self.inputted_text[-1] == ' ':
            return
        if self.WORD_CORRECTION == self.CORRECT_WORD:
            tags = self.inputted_text.split(' ')
            if len(tags) > 0 and tags[-1] != '':
                word = self.decoder.predict(self.inputted_data[-len(tags[-1]):], self.task[:len(self.inputted_text)])
                if word != '': # '' means no match
                    tags[-1] = word
                self.inputted_text = ' '.join(tags)
        if i < len(self.task):
            self.inputted_text += ' '
            self.inputted_data.append(input_data)
    
    def delete_a_letter(self):
        if len(self.inputted_text) > 0:
            self.inputted_text = self.inputted_text[:-1]
            self.inputted_data = self.inputted_data[:-1]

    def get_position(self, data): # get position from inputted data
        [side, index, highlight_row, highlight_col] = data[:4]
        row = max(0-0.5,min(2+0.5,highlight_row - 1))
        col = max(0-0.5,min(1+0.5,highlight_col - 1))
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
        return [col, row]
        