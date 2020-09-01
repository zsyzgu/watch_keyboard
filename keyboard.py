import cv2
import numpy as np
import time
import random
import pygame
import math

class Keyboard:
    GRID = 50
    
    def __init__(self):
        self.VISABLE_FEEDBACK = False
        self.WORD_CORRECTION = False
        self.init_letter_positions()
        self.init_task_list('phrases.txt')
        self.inputted_text = ''
        self.init_candidates()
        self.init_display()
    
    def init_candidates(self):
        self.candidates = ['' for i in range(5)]
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
                #self.letter_positions[index] = [c + r / 2, r]
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
    
    def init_display(self):
        self.screen = pygame.display.set_mode((10 * self.GRID + 1, 5 * self.GRID + 1))
        pygame.display.set_caption('Qwerty Watch')
        self.hl_L_row = None # Hightline line
        self.hl_L_col = None
        self.hl_R_row = None
        self.hl_R_col = None

    def draw(self):
        GRID = self.GRID
        image = np.zeros((5 * GRID + 1, 10 * GRID + 1, 3), np.uint8)

        cv2.rectangle(image, (0, 0), (10 * GRID, GRID - 1), (0, 0, 0), -1)

        # Draw task and inputted text
        cv2.putText(image, self.task_list[self.curr_task_id], (int(GRID * 0.5), int(GRID * 0.4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(image, self.inputted_text + '_', (int(GRID * 0.5), int(GRID * 0.8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Draw candidates
        for i in range(5):
            cv2.rectangle(image, (i * 2 * GRID, GRID), ((i + 1) * 2 * GRID, 2 * GRID), (255, 255, 255), 1)
            candidate = self.candidates[i]
            font_size = 0.6
            if (len(candidate) >= 8):
                font_size = 4.8 / len(candidate)
            cv2.putText(image, candidate, (int(i * 2 * GRID) + 5, 2 * GRID - 10), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 1)

        # Draw the keyboard layout
        for i in range(26):
            ch = chr(i + ord('A'))
            pos = self.letter_positions[i]
            bg_color = self.letter_colors[i]
            cv2.rectangle(image, (int(pos[0] * GRID), int((pos[1] + 2) * GRID)), (int((pos[0] + 1) * GRID), int((pos[1] + 3) * GRID)), bg_color, -1)
            cv2.rectangle(image, (int(pos[0] * GRID), int((pos[1] + 2) * GRID)), (int((pos[0] + 1) * GRID), int((pos[1] + 3) * GRID)), (255, 255, 255), 1)
            cv2.putText(image, ch, (int(pos[0] * GRID) + 15, int((pos[1] + 3) * GRID) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Hightlight Row
        if self.VISABLE_FEEDBACK:
            if self.hl_L_row != None:
                row = max(0.5, min(3.5, self.hl_L_row))
                row_pixel = int((row - 0.5 + 2) * GRID)
                image[row_pixel-2:row_pixel+3,:5*GRID] *= 2
            if self.hl_R_row != None:
                row = max(0.5, min(3.5, self.hl_R_row))
                row_pixel = int((row - 0.5 + 2) * GRID)
                image[row_pixel-2:row_pixel+3,5*GRID:] *= 2
            if self.hl_L_col != None:
                col = max(0.5, min(2.5, self.hl_L_col))
                col_pixel = int((2.5 + col) * GRID)
                image[2*GRID:5*GRID,col_pixel-2:col_pixel+3] *= 2
            if self.hl_R_col != None:
                col = max(0.5, min(2.5, self.hl_R_col))
                col_pixel = int((7.5 - col) * GRID)
                image[2*GRID:5*GRID,col_pixel-2:col_pixel+3] *= 2

        pg_img = pygame.surfarray.make_surface(cv2.transpose(image))
        self.screen.blit(pg_img, (0,0))
        pygame.display.flip()

    def update_hightlight(self, hl_L_row, hl_L_col, hl_R_row, hl_R_col):
        if hl_L_row != None:
            self.hl_L_row = hl_L_row
        if hl_L_col != None:
            self.hl_L_col = hl_L_col
        if hl_R_row != None:
            self.hl_R_row = hl_R_row
        if hl_R_col != None:
            self.hl_R_col = hl_R_col
    
    def next_phrase(self):
        self.curr_task_id += 1
        print('Phase = %d' % (self.curr_task_id))
        self.inputted_text = ''
        self.init_candidates()
        if self.curr_task_id >= len(self.task_list):
            self.curr_task_id = 0
            return False
        return True

    def redo_phrase(self):
        self.inputted_text = ''
        self.init_candidates()

    def enter_a_letter(self, default = None):
        i = len(self.inputted_text)
        task = self.task_list[self.curr_task_id]
        letter = ''
        if i < len(task) and task[i] != ' ':
            if default == None:
                letter = task[i]
            else:
                letter = default
            self.inputted_text += letter
        return letter

    def enter_a_space(self):
        i = len(self.inputted_text)
        task = self.task_list[self.curr_task_id]
        if self.WORD_CORRECTION and (i >= len(task) - 1 or task[i] == ' '):
            tags = self.inputted_text.split(' ')
            if len(tags) > 0 and len(tags[-1]) > 0:
                word = tags[-1]
                word = self.word_correction(word)
                tags[-1] = word
                self.inputted_text = ' '.join(tags)
        if i < len(task) and task[i] == ' ':
            self.inputted_text += ' '
    
    def delete_a_word(self):
        if len(self.inputted_text) > 0:
            words = self.inputted_text.split()
            if len(words) == 1:
                self.inputted_text = ''
            else:
                self.inputted_text = ' '.join(words[:-1]) + ' '
        self.init_candidates()
    
    def delete_a_letter(self):
        if len(self.inputted_text) > 0:
            self.inputted_text = self.inputted_text[:-1]
    
    def enter_a_word(self, word):
        if len(self.inputted_text) > 0 and len(word) > 0:
            self.inputted_text += ' '
        self.inputted_text += word
        self.init_candidates()

    def update_candidates(self, candidates):
        self.candidates = candidates.copy()

    def word_correction(self, word):
        positions = []
        for letter in word:
            index = ord(letter) - ord('a')
            positions.append(self.letter_positions[index])
        
        N = len(self.corpus)
        max_pri = 0
        best_candidate = word
        for i in range(N):
            (candidate, pri) = self.corpus[i]
            if len(candidate) == len(word):
                for j in range(len(word)):
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
            
