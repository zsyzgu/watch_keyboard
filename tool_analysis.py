import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from astroML.stats import fit_bivariate_normal
from astroML.stats.random import bivariate_normal
import math
import os
import time
import cv2
from finger_tracker import FingerTracker
from decoder import Decoder

class Simulation:
    def __init__(self):
        pass
    
    def input(self):
        nums = sys.argv[1].split('-')
        assert(len(nums) == 2)
        if nums[0].isdigit():
            users = [int(nums[0])]
        else:
            users = range(1, 13)
        if nums[1].isdigit():
            sessions = [int(nums[1])]
        else:
            sessions = range(1, 6)

        N = 20
        task_list = []
        inputted_list = []
        data_list = []
        space_cnt_list = []

        for user in users:
            for session in sessions:
                folder_path = 'data-baseline/' + str(user) + '-' + str(session) + '/'
                for i in range(N):
                    file_path = folder_path + str(i) + '.pickle'
                    if os.path.exists(file_path):
                        print(file_path)
                        [task, inputted, data, space_cnt] = pickle.load(open(file_path, 'rb'))
                        assert(len(inputted) == len(data) and len(data) == len(data))
                        task_list.append(task)
                        inputted_list.append(inputted)
                        data_list.append(data)
                        space_cnt_list.append(space_cnt)
        
        return task_list, inputted_list, data_list, space_cnt_list

    def run(self):
        task_list, inputted_list, data_list, space_cnt_list = self.input()

        wpm_word_cnt = 0
        word_cnt = 0
        correct_word_cnt = 0
        inputted_word_cnt = 0
        minute_cnt = 0

        for task, inputted, data, space_cnt in zip(task_list, inputted_list, data_list, space_cnt_list):
            words = task.split()
            begin = 0
            for word in words:
                end = begin + len(word)
                enter = inputted[begin:end]
                word_data = data[begin:end]
                if enter == word:
                    correct_word_cnt += 1
                begin = end + 1
            wpm_word_cnt += (len(task) - 1) / 5.0
            minute_cnt += (float(data[-1][4]) - float(data[0][4])) / 60
            word_cnt += len(words)
            inputted_word_cnt += space_cnt + 1
            break
        
        wpm = wpm_word_cnt / minute_cnt
        uer = 1 - float(correct_word_cnt) / word_cnt
        cer = float(inputted_word_cnt) / word_cnt - 1
        print('User = %s, WPM = %f, UER = %f, CER = %f' % (sys.argv[1].split('-')[0], wpm, uer, cer))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('[Usage] python tool_analysis.py folder_name(x-x)')
        exit()

    users = [1]
    for user in users:
        sys.argv[1] = str(user) + '-x'
        Simulation().run()