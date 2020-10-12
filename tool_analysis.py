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

class Simulation:
    def __init__(self):
        self.init_corpus()
    
    def init_corpus(self):
        self.corpus = []
        lines = open('corpus.txt').readlines()
        for i in range(20000):
            tags = lines[i].split(' ')
            word = tags[0]
            pri = float(tags[1])
            self.corpus.append([word, pri])
    
    def calc_letter_distribution(self, **kwargs):
        data_list = kwargs['data_list']
        task_list = kwargs['task_list']
        assert(len(data_list) == len(task_list))

        QWERTY = ['QWERTYUIOP', 'ASDFGHJKL', 'ZXCVBNM']
        self.letter_positions = [[-1, -1] for i in range(26)]
        self.letter_fingers = np.zeros((26, 10))
        self.letter_distributions = [[[-1, -1, 0.1, 0.1, 0.1, 0] for finger in range(10)] for alpha in range(26)] # Formal = [xc, yc, std_x2, std_y2, std_xy, p]
        for r in range(3):
            line = QWERTY[r]
            for c in range(len(line)):
                ch = line[c]
                alpha = ord(ch) - ord('A')
                self.letter_positions[alpha] = [c, r]

        features = [[[] for finger in range(10)] for alpha in range(26)]
        for data, task in zip(data_list, task_list):
            assert(len(data) == len(task))
            for i in range(len(task)):
                letter = task[i]
                if letter.isalpha():
                    alpha = ord(letter) - ord('a')
                    feature = self.get_feature(data[i])
                    finger = self.get_finger(data[i])
                    features[alpha][finger].append(feature)
        
        for alpha in range(26):
            for finger in range(10):
                points = np.array(features[alpha][finger])
                if len(points) >= 1:
                    self.letter_fingers[alpha][finger] += len(points)
                    X = points[:, 0]
                    Y = points[:, 1]
                
                    if len(points) >= 5: # Remove > 3_std
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
                    
                    xc = np.mean(X)
                    yc = np.mean(Y)
                    
                    cov = np.array([[0.01, 0], [0, 0.01]])
                    if len(points) >= 5:
                        cov = np.cov(np.array([X,Y]))
                    
                    std_x2 = cov[0, 0]
                    std_y2 = cov[1, 1]
                    std_xy = (std_x2 ** 0.5) * (std_y2 ** 0.5)
                    p = cov[0, 1] / std_xy
                    assert(not(np.isnan(std_x2) or np.isnan(std_y2) or np.isnan(std_xy)))
                    self.letter_distributions[alpha][finger] = [xc, yc, std_x2, std_y2, std_xy, p]
            
            if sum(self.letter_fingers[alpha]) != 0:
                self.letter_fingers[alpha] /= sum(self.letter_fingers[alpha])
                std_fingering = np.argmax(self.letter_fingers[alpha])
                for finger in range(10):
                    if self.letter_fingers[alpha][finger] == 0:
                        self.letter_distributions[alpha][finger] = self.letter_distributions[alpha][std_fingering].copy()
                    self.letter_fingers[alpha][finger] = max(self.letter_fingers[alpha][finger], 0.01)
    
    def get_finger(self, data):
        [side, finger, highlight_row, highlight_col, timestamp, palm_line, endpoint_x, endpoint_y, frame_id] = data
        assert(side == 'L' or side == 'R')
        if side == 'L':
            return finger
        if side == 'R':
            return finger + 5

    def get_feature(self, data): # get position from inputted data
        [side, finger, highlight_row, highlight_col, timestamp, palm_line, endpoint_x, endpoint_y, frame_id] = data
        row = max(0-0.5,min(2+0.5,highlight_row - 1)) # Display setting
        col = max(0-0.5,min(1+0.5,highlight_col - 1))
        if side == 'L':
            if finger == 1:
                col = 3 + col
            else:
                col = 3 - (finger - 1)
        if side == 'R':
            if finger == 1:
                col = 6 - col
            else:
                col = 6 + (finger - 1)
            
            col += 1
            endpoint_x += 10
        #return [col, row]
        return [endpoint_x, endpoint_y]

    def get_position(self, data):
        [side, finger, highlight_row, highlight_col, timestamp, palm_line, endpoint_x, endpoint_y, frame_id] = data
        row = int(round(max(0,min(2,highlight_row - 1))))
        col = int(round(max(0,min(1,highlight_col - 1))))
        if side == 'L':
            if finger == 1:
                col = 3 + col
            else:
                col = 3 - (finger - 1)
        if side == 'R':
            if finger == 1:
                col = 6 - col
            else:
                col = 6 + (finger - 1)
        return [col, row]

    def predict(self, data, truth):
        features = [self.get_feature(tap) for tap in data]
        fingers = [self.get_finger(tap) for tap in data]
        positions = [self.get_position(tap) for tap in data]

        P = np.zeros((len(features), 26)) # P(alpha | w_i)
        for i in range(len(features)):
            for alpha in range(26):
                finger = fingers[i]
                [x, y] = features[i]
                [xc, yc, std_x2, std_y2, std_xy, p] = self.letter_distributions[alpha][finger]
                dx = x - xc
                dy = y - yc
                z = (dx ** 2) / std_x2 - (2 * p * dx * dy) / std_xy + (dy ** 2) / std_y2
                step_prob = self.letter_fingers[alpha][finger]
                step_prob *= (.001 / (std_xy * ((1 - p ** 2) ** 0.5))) * math.exp(-z / (2 * (1 - p ** 2))) # the constant is modified to be small (1/2pi --> .01) so that prob<1
                assert(step_prob < 1)
                P[i, alpha] = step_prob

        max_prob = 0
        best_candidate = ''
        probs = []
        for (candidate, prob) in self.corpus:
            if len(candidate) == len(features):
                for i in range(len(features)):
                    letter = candidate[i]
                    alpha = ord(letter) - ord('a')

                    [xc, yc] = self.letter_positions[alpha]
                    [x, y] = positions[i]
                    if (x - xc) ** 2 + (y - yc) ** 2 >= 2.1 ** 2:
                        prob = 0
                        break

                    step_prob = P[i, alpha]
                    prob *= step_prob
                    if prob < max_prob:
                        break

                if prob > max_prob:
                    max_prob = prob
                    best_candidate = candidate
                probs.append(prob)
                if candidate == truth:
                    truth_prob = prob
        
        rank = 1 + sum(np.array(probs) > truth_prob)
        if truth_prob == 0:
            rank = -1
        return best_candidate, rank

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

        for user in users:
            for session in sessions:
                folder_path = 'data/' + str(user) + '-' + str(session) + '/'
                for i in range(N):
                    file_path = folder_path + str(i) + '.pickle'
                    if os.path.exists(file_path):
                        [task, inputted, data] = pickle.load(open(file_path, 'rb'))
                        assert(len(inputted) == len(data) and len(data) == len(data))
                        task_list.append(task)
                        inputted_list.append(inputted)
                        data_list.append(data)
        
        return task_list, inputted_list, data_list

    def run(self):
        task_list, inputted_list, data_list = self.input()
        self.calc_letter_distribution(data_list=data_list, task_list=task_list)

        ranks = []
        fail_cases = []
        for task, inputted, data in zip(task_list, inputted_list, data_list):
            words = task.split()
            begin = 0
            for word in words:
                end = begin + len(word)
                
                enter = inputted[begin:end]
                word_data = data[begin:end]
                if enter == word:
                    pred, rank = self.predict(word_data, word)
                    ranks.append(rank)
                    if pred != word and rank != -1:
                        #print('[Fail Cases]', word, pred, rank)
                        fail_cases.append([word, word_data])

                begin = end + 1

        print('=====   Top-5 accuracy   =====')
        ranks = np.array(ranks)
        probs = []
        for i in range(5):
            #prob = sum(ranks == i+1) / len(ranks)
            prob = sum(ranks == i+1) / sum(ranks != -1)
            print('Rank %d = %f' % (i+1, prob))
            if i == 0:
                TOP_1 = prob
        
        pickle.dump([self, fail_cases], open('debug.pickle', 'wb'))
        return TOP_1

class Debug:
    def __init__(self):
        pass

    def run(self):
        [sim, fail_cases] = pickle.load(open('debug.pickle', 'rb'))

        for [word, data] in fail_cases:
            plt.scatter(features[:,0], features[:,1])
            plt.show()
            pred, rank = sim.predict(data, word)
            print(word, pred)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('[Usage] python tool_analysis.py folder_name(x-x)')
        exit()
    if sys.argv[1] != 'x-x':
        Simulation().run()
    else:
        users = [1,2,3,4,5,6,8,9,10,12]
        top_1 = []
        for user in users:
            print('User = %d' % user)
            sys.argv[1] = str(user) + '-x'
            prob = Simulation().run()
            top_1.append(prob)
        print(np.mean(top_1))
