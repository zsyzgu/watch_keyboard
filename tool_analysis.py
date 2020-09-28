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
    LETTER_DISTRIBUTION_LAYOUT = 0
    LETTER_DISTRIBUTION_STAT = 1

    def __init__(self):
        self.init_corpus()
        self.calc_letter_distribution()
    
    def init_corpus(self):
        self.corpus = []
        lines = open('corpus.txt').readlines()
        for i in range(20000):
            tags = lines[i].split(' ')
            word = tags[0]
            pri = float(tags[1])
            self.corpus.append([word, pri])
    
    def calc_letter_distribution(self, mode = LETTER_DISTRIBUTION_LAYOUT, **kwargs):
        if mode == self.LETTER_DISTRIBUTION_LAYOUT: # set letter distribution according to keyboard layout
            QWERTY = ['QWERTYUIOP', 'ASDFGHJKL', 'ZXCVBNM']
            self.letter_distributions = [[-1, -1, 0.1, 0.1, 0.1, 0] for i in range(26)] # Formal = [xc, yc, std_x2, std_y2, std_xy, p]
            for r in range(3):
                line = QWERTY[r]
                for c in range(len(line)):
                    ch = line[c]
                    index = ord(ch) - ord('A')
                    self.letter_distributions[index][:2] = [c, r]
        if mode == self.LETTER_DISTRIBUTION_STAT: # set letter distribution according to statistic analysis
            data_list = kwargs['data_list']
            task_list = kwargs['task_list']
            points = [[] for ch in range(26)]

            assert(len(data_list) == len(task_list))
            for data, task in zip(data_list, task_list):
                assert(len(data) == len(task))
                for i in range(len(task)):
                    letter = task[i]
                    if letter.isalpha():
                        index = ord(letter) - ord('a')
                        position = self.get_position(data[i])
                        points[index].append(position)
            
            fig, ax = plt.subplots()
            ans = []
            for index in range(26):
                if len(points[index]) >= 2:
                    X = np.array(points[index])[:,0]
                    Y = np.array(points[index])[:,1]
                    n_std = 3

                    (center, a, b, theta) = fit_bivariate_normal(X, Y, robust=False)
                    xc, yc = center
                    ell = Ellipse(center, a * n_std, b * n_std, (theta * 180. / np.pi), ec='k', fc='none', color='red')

                    # TODO: exclude >3_std
                    '''
                    pack = zip(X.copy(), Y.copy())
                    X = []
                    Y = []
                    for x, y in pack:
                        if ell.contains_point([x, y]):
                            X.append(x)
                            Y.append(y)
                    (center, a, b, theta) = fit_bivariate_normal(X, Y, robust=False)
                    xc, yc = center
                    ell = Ellipse(center, a * n_std, b * n_std, (theta * 180. / np.pi), ec='k', fc='none', color='red')
                    '''

                    plt.scatter(X, Y, color=('C'+str(index)), s = 5)
                    plt.scatter(xc, yc, color='red', s = 10)
                    ax.add_patch(ell)

                    cov = np.cov(np.array([X,Y]))
                    cov[0,0] = max(0.01,cov[0,0])
                    cov[1,1] = max(0.01,cov[1,1])
                    std_x2 = cov[0, 0]
                    std_y2 = cov[1, 1]
                    std_xy = (std_x2 ** 0.5) * (std_y2 ** 0.5)
                    p = cov[0, 1] / std_xy
                    self.letter_distributions[index] = [xc, yc, std_x2, std_y2, std_xy, p]
                else:
                    print('lack of', chr(index + ord('a')))
            plt.show()
    
    def get_position(self, data): # get position from inputted data
        [side, index, highlight_row, highlight_col, timestamp, palm_line, endpoint_x, endpoint_y, frame_id] = data
        #row = max(0-0.5,min(2+0.5,highlight_row - 1)) # Display setting
        #col = max(0-0.5,min(1+0.5,highlight_col - 1))
        row = max(0-1,min(2+1,highlight_row - 1)) # Decode setting
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
            col += 1
        return [col, row]
        #return [endpoint_x, endpoint_y]

    def predict(self, positions, truth):
        N = len(self.corpus)
        max_pri = 0
        best_candidate = ''
        pris = []
        for i in range(N):
            (candidate, pri) = self.corpus[i]
            if len(candidate) == len(positions):
                for j in range(len(positions)):
                    letter = candidate[j]
                    index = ord(letter) - ord('a')
                    [x, y] = positions[j]
                    [xc, yc, std_x2, std_y2, std_xy, p] = self.letter_distributions[index]
                    dx = x - xc
                    dy = y - yc
                    z = (dx ** 2) / std_x2 - (2 * p * dx * dy) / std_xy + (dy ** 2) / std_y2
                    prob = (.01 / (std_xy * ((1 - p ** 2) ** 0.5))) * math.exp(-z / (2 * (1 - p ** 2))) # the constant is modified to be small (1/2pi --> .01) so that prob<1
                    pri *= prob
                    assert(prob < 1)
                    if pri < max_pri:
                        break
                if pri > max_pri:
                    max_pri = pri
                    best_candidate = candidate

                pris.append(pri)
                if candidate == truth:
                    truth_pri = pri
        
        rank = 1
        for pri in pris:
            if pri > truth_pri:
                rank += 1

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
        self.calc_letter_distribution(mode=self.LETTER_DISTRIBUTION_STAT, data_list=data_list, task_list=task_list)
        
        tracker_L = FingerTracker(1)
        tracker_R = FingerTracker(2)

        ranks = []
        fail_cases = []
        ts = []
        for task, inputted, data in zip(task_list, inputted_list, data_list):
            words = task.split()
            begin = 0
            for word in words:
                end = begin + len(word)
                
                enter = inputted[begin:end]
                word_data = data[begin:end]
                points = [self.get_position(w) for w in word_data]
                if enter == word:
                    t=time.clock()
                    pred, rank = self.predict(points, word)
                    t=time.clock()-t
                    ts.append(t)
                    ranks.append(rank)
                    if pred != word:
                        print('[Fail Cases]', word, pred)
                        fail_cases.append([word, word_data])

                        '''
                        for index in range(len(word_data)):
                            d = word_data[index]
                            [side, which_finger, highlight_row, highlight_col, timestamp, palm_line, endpoint_x, endpoint_y, frame_id] = d
                            print(word, word[index], frame_id)
                            frame_L = cv2.imread('data/' + sys.argv[1] + '/L/' + str(frame_id) + '.jpg')
                            frame_R = cv2.imread('data/' + sys.argv[1] + '/R/' + str(frame_id) + '.jpg')
                            tracker_L.run(frame_L)
                            tracker_R.run(frame_R)
                            output_L = tracker_L.output()
                            output_R = tracker_R.output()
                            output = np.hstack([output_L, output_R])
                            if side == 'L':
                                record = tracker_L.palm_line
                            else:
                                record = tracker_R.palm_line
                            if abs(record - palm_line) > 10:
                                print('[Palm Line Chaged]', record, palm_line)
                            cv2.imshow('illustration', output)
                            cv2.waitKey(0)
                        '''

                begin = end + 1
        print(np.mean(ts))

        print('=====   Top-5 accuracy   =====')
        ranks = np.array(ranks)
        for rank in range(1, 6):
            print(sum(ranks == rank) / len(ranks))
        
        pickle.dump([self, fail_cases], open('debug.pickle', 'wb'))

class Debug:
    def __init__(self):
        pass

    def run(self):
        [sim, fail_cases] = pickle.load(open('debug.pickle', 'rb'))

        for [word, data] in fail_cases:
            points = np.array([sim.get_position(d) for d in data])
            plt.scatter(points[:,0], points[:,1])
            plt.show()
            pred, rank = sim.predict(points, word)
            print(word, pred)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('[Usage] python tool_analysis.py folder_name(x-x)')
        exit()
    Simulation().run()
    #Debug().run()
