import pygame
import time
from keyboard import Keyboard

class Baseline:
    def __init__(self):
        pygame.mixer.init(22050, -16, 2, 64)
        pygame.init()
        self.keyboard = Keyboard(VISABLE_FEEDBACK=Keyboard.VISABLE_NO, WORD_CORRECTION=Keyboard.CORRECT_WORD)

    def save_data(self):
        pass
        # TODO

    def run(self):
        self.keyboard.draw()

        is_running = True
        while is_running:
            keys = []
            is_mouse_down = False
            mouse_pos = None
            for event in pygame.event.get():
                if event.type == pygame.KEYUP:
                    keys.append(event.key)
                if event.type == pygame.MOUSEBUTTONDOWN:
                    is_mouse_down = True
                    mouse_pos = pygame.mouse.get_pos()
            
            inputted = ''
            inputted_pos = [0, 0]
            if is_mouse_down:
                pos_list = self.keyboard.decoder.positions.copy()
                chr_list = [chr(i + ord('a')) for i in range(26)]
                pos_list.extend([[7,2], [8,2], [9,1], [9,2]])
                chr_list.extend([' ', ' ', '-', '-'])
                GRID = self.keyboard.GRID
                [x, y] = mouse_pos
                for (pos, ch) in zip(pos_list, chr_list):
                    [x0, x1] = [int((pos[0] + 0) * GRID), int((pos[0] + 1) * GRID)]
                    [y0, y1] = [int((pos[1] + 1) * GRID), int((pos[1] + 2) * GRID)]
                    if x0 <= x and x < x1 and y0 <= y and y < y1:
                        inputted = ch
                        inputted_pos = pos
                        break

                if inputted_pos[0] <= 4:
                    side = 'L'
                    finger = max(1, 4 - inputted_pos[0])
                else:
                    side = 'R'
                    finger = max(1, inputted_pos[0] - 5)
                corr_endpoint_x = inputted_pos[0] - 3
                highlight_row = inputted_pos[1] + 1

                # [side, which_finger, highlight_row, highlight_col, timestamp, palm_line, endpoint_x, endpoint_y, corr_endpoint_x, corr_endpoint_y, image_L, image_R]
                inputted_data = [side, finger, highlight_row, 0, time.clock(), 0, 0, 0, corr_endpoint_x, 0, 0, 0]
                if inputted.isalpha():
                    self.keyboard.enter_a_letter(inputted_data, inputted)
                elif inputted == ' ':
                    self.keyboard.enter_a_space(inputted_data)
                elif inputted == '-':
                    self.keyboard.delete_a_letter()
            
            if pygame.K_q in keys:
                is_running = False
            
            if pygame.K_n in keys:
                if len(self.keyboard.inputted_text) == len(self.keyboard.task):
                    self.keyboard.enter_a_space(inputted_data) # Correct and show the last word
                    self.keyboard.draw()
                    start_time = float(self.keyboard.inputted_data[0][4])
                    end_time = float(self.keyboard.inputted_data[-1][4])
                    wpm = ((len(self.keyboard.inputted_text)-1)/5.0)/((end_time - start_time)/60.0)
                    print('WPM = ', wpm)
                    self.save_data()
                    is_running = self.keyboard.next_phrase()
            
            if pygame.K_r in keys: # Redo phrase
                self.keyboard.redo_phrase()
            
            self.keyboard.draw()
            

if __name__ == "__main__":
    Baseline().run()
