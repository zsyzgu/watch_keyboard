import cv2
import numpy as np
import time
import random
import pygame
import math
from PIL import Image, ImageFont, ImageDraw

class Keyboard:
    GRID = 50
    VISABLE_NO = 0
    VISABLE_TOUCH = 1
    VISABLE_ALWAYS = 2
    CORRECT_NO = 0
    CORRECT_WORD = 1
    CORRECT_LETTER = 2
    
    def __init__(self, VISABLE_FEEDBACK = VISABLE_ALWAYS, WORD_CORRECTION = CORRECT_WORD):
        self.VISABLE_FEEDBACK = VISABLE_FEEDBACK
        self.WORD_CORRECTION = WORD_CORRECTION
        self.init_letter_positions()
        self.init_task_list('phrases.txt')
        self.init_corpus()
        self.init_inputted_data()
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

    def init_task_list(self, path):
        self.task_list = []
        self.curr_task_id = 0

        lines = open(path).readlines()[0:20]
        for line in lines:
            line = line.lower()
            self.task_list.append(line.strip('\n'))

        random.shuffle(self.task_list)

    def init_corpus(self):
        if self.WORD_CORRECTION:
            self.corpus = []
            lines = open('corpus.txt').readlines()
            CORPUS_AMOUNT = 20000
            for i in range(CORPUS_AMOUNT):
                line = lines[i]
                tags = line.strip().split(' ')
                word = tags[0]
                pri = int(tags[1])
                self.corpus.append((word, pri))

    def init_inputted_data(self):
        self.inputted_text = ''
        self.inputted_data = []

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
        cv2.putText(image, self.task_list[self.curr_task_id], (int(GRID * 0.5), int(GRID * 0.4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(image, self.inputted_text + '_', (int(GRID * 0.5), int(GRID * 0.8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Draw the keyboard layout
        for i in range(26):
            ch = chr(i + ord('A'))
            pos = self.letter_positions[i]
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
            pass
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
        return True

    def redo_phrase(self):
        self.inputted_text = ''
        self.inputted_data = []

    def enter_a_letter(self, input_data, input_letter):
        i = len(self.inputted_text)
        task = self.task_list[self.curr_task_id]
        letter = ''
        if i < len(task):
            if self.WORD_CORRECTION == self.CORRECT_LETTER:
                letter = task[i]
            else:
                letter = input_letter
            self.inputted_text += letter
            self.inputted_data.append(input_data)
        return letter

    def enter_a_space(self, input_data):
        i = len(self.inputted_text)
        task = self.task_list[self.curr_task_id]
        if self.WORD_CORRECTION == self.CORRECT_WORD:
            tags = self.inputted_text.split(' ')
            if len(tags) > 0 and tags[-1] != '':
                word = self.word_correction(self.inputted_data[-len(tags[-1]):])
                assert(len(tags[-1]) == len(word))
                tags[-1] = word
                self.inputted_text = ' '.join(tags)
        if i < len(task):
            self.inputted_text += ' '
            self.inputted_data.append(input_data)
    
    def delete_a_letter(self):
        if len(self.inputted_text) > 0:
            self.inputted_text = self.inputted_text[:-1]
            self.inputted_data = self.inputted_data[:-1]

    def word_correction(self, inputted_data):
        positions = []
        for data in inputted_data:
            [side, index, highlight_row, highlight_col] = data[:4]
            row = max(0-1,min(2+1,highlight_row - 1))
            col = max(0-1,min(1+1,highlight_col - 1))
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

            positions.append([col, row])
        
        N = len(self.corpus)
        max_pri = 0
        best_candidate = ''
        for i in range(N):
            (candidate, pri) = self.corpus[i]
            if len(candidate) == len(inputted_data):
                for j in range(len(inputted_data)):
                    letter = candidate[j]
                    index = ord(letter) - ord('a')
                    position = self.letter_positions[index]
                    dx = position[0] - positions[j][0]
                    dy = position[1] - positions[j][1]
                    pri *= math.exp(-((2*dx)**2+(2*dy)**2))
                if pri > max_pri:
                    max_pri = pri
                    best_candidate = candidate
        
        return best_candidate
            
