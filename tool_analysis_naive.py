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
        [self.bgrams_index, self.bgrams_freq] = pickle.load(open('2grams.model', 'rb'))
        [self.tgrams_index, self.tgrams_freq] = pickle.load(open('3grams.model', 'rb'))
    
    def init_corpus(self):
        self.size = 20000
        self.corpus = []
        self.corpus_map = {}
        self.corpus_prob = []
        lines = open('corpus.txt').readlines()
        for i in range(20000):
            tags = lines[i].split(' ')
            word = tags[0]
            prob = float(tags[1])
            self.corpus.append(word)
            self.corpus_prob.append(prob)
            self.corpus_map[word] = i
    
    def calc_letter_distribution(self, mode = LETTER_DISTRIBUTION_LAYOUT, **kwargs):
        if mode == self.LETTER_DISTRIBUTION_LAYOUT: # set letter distribution according to keyboard layout
            QWERTY = ['QWERTYUIOP', 'ASDFGHJKL', 'ZXCVBNM']
            self.letter_positions = [[-1, -1] for i in range(26)]
            self.letter_distributions = [[-1, -1, 0.1, 0.1, 0.1, 0] for i in range(26)] # Formal = [xc, yc, std_x2, std_y2, std_xy, p]
            for r in range(3):
                line = QWERTY[r]
                for c in range(len(line)):
                    ch = line[c]
                    index = ord(ch) - ord('A')
                    self.letter_positions[index] = [c, r]
                    self.letter_distributions[index][:2] = [c, r]
        if mode == self.LETTER_DISTRIBUTION_STAT: # set letter distribution according to statistic analysis
            data_list = kwargs['data_list']
            task_list = kwargs['task_list']

            assert(len(data_list) == len(task_list))

            features = [[] for ch in range(26)]
            for data, task in zip(data_list, task_list):
                assert(len(data) == len(task))
                for i in range(len(task)):
                    letter = task[i]
                    if letter.isalpha():
                        index = ord(letter) - ord('a')
                        feature = self.get_feature(data[i])
                        features[index].append(feature)
            
            fig, ax = plt.subplots()
            ans = []
            for index in range(26):
                if len(features[index]) >= 2:
                    X = np.array(features[index])[:,0]
                    Y = np.array(features[index])[:,1]
                    n_std = 3

                    if len(features[index]) >= 5:
                        # exclude >3_std
                        xc, x_std = np.mean(X), np.std(X)
                        yc, y_std = np.mean(Y), np.std(Y)
                        pack = zip(X.copy(), Y.copy())
                        X = []
                        Y = []
                        for x, y in pack:
                            if abs(x-xc) <= n_std * x_std and abs(y-yc) <= n_std * y_std:
                                X.append(x)
                                Y.append(y)

                        (center, a, b, theta) = fit_bivariate_normal(X, Y, robust=False)
                        xc, yc = center
                        ell = Ellipse(center, a * n_std, b * n_std, (theta * 180. / np.pi), ec='k', fc='none', color='red')

                        plt.scatter(X, Y, color=('C'+str(index)), s = 5)
                        plt.scatter(xc, yc, color='red', s = 10)
                        ax.add_patch(ell)

                    
                    cov = np.array([[0.1, 0], [0, 0.1]])
                    if len(X) >= 5:
                        cov = np.cov(np.array([X,Y]))
                    std_x2 = cov[0, 0]
                    std_y2 = cov[1, 1]
                    std_xy = (std_x2 ** 0.5) * (std_y2 ** 0.5)
                    p = cov[0, 1] / std_xy
                    self.letter_distributions[index] = [xc, yc, std_x2, std_y2, std_xy, p]
                else:
                    print('Lack =', chr(index + ord('a')))
            #plt.show()
    
    def get_feature(self, data): # get position from inputted data
        [side, index, highlight_row, highlight_col, timestamp, palm_line, endpoint_x, endpoint_y, corr_endpoint_x, corr_endpoint_y] = data[:10]
        row = max(0-0.5,min(2+0.5,highlight_row - 1)) # Display setting
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
            
            col += 1
            endpoint_x += 11
        return [corr_endpoint_x, highlight_row]

    def get_position(self, data):
        [side, index, highlight_row, highlight_col, timestamp, palm_line, endpoint_x, endpoint_y, corr_endpoint_x, corr_endpoint_y] = data[:10]
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
        return [col, row]

    def binary_search(self, arr, key):
        st = 0
        en = len(arr) - 1
        while (st < en):
            mid = (st + en + 1) // 2 # TODO: check
            if key >= arr[mid]:
                st = mid
            else:
                en = mid - 1
        if arr[st] == key:
            return st
        else:
            return -1

    def get_corpus_prob(self, inputted):
        words = inputted.split()
        index = -1
        if len(words) >= 3 and words[-3] in self.corpus_map and words[-2] in self.corpus_map: # Trigram
            key = self.corpus_map[words[-3]] * self.size + self.corpus_map[words[-2]]
            index = self.binary_search(self.tgrams_index[:,0], key)
            ngrams_index = self.tgrams_index
            ngrams_freq = self.tgrams_freq
        if index == -1 and len(words) >= 2 and words[-2] in self.corpus_map: # Bigram
            key = self.corpus_map[words[-2]]
            index = self.binary_search(self.bgrams_index[:,0], key)
            ngrams_index = self.bgrams_index
            ngrams_freq = self.bgrams_freq
        if index != -1:
            st = ngrams_index[index, 1]
            if index + 1 < ngrams_index.shape[0]:
                en = ngrams_index[index + 1, 1]
            else:
                en = ngrams_freq.shape[0]
            corpus_prob = np.zeros(self.size)
            for i in range(st, en):
                corpus_prob[ngrams_freq[i, 0]] = ngrams_freq[i, 1]
            corpus_prob[corpus_prob == 0] = 1
            return corpus_prob
        return self.corpus_prob # Unigram

    def predict(self, data, inputted, truth):
        features = [self.get_feature(tap) for tap in data]
        positions = [self.get_position(tap) for tap in data]

        P = np.zeros((len(features), 26)) # P(alpha | w_i)
        for i in range(len(features)):
            for index in range(26):
                [x, y] = features[i]
                [xc, yc, std_x2, std_y2, std_xy, p] = self.letter_distributions[index]
                dx = x - xc
                dy = y - yc
                z = (dx ** 2) / std_x2 - (2 * p * dx * dy) / std_xy + (dy ** 2) / std_y2
                step_prob = (.001 / (std_xy * ((1 - p ** 2) ** 0.5))) * math.exp(-z / (2 * (1 - p ** 2))) # the constant is modified to be small (1/2pi --> .01) so that prob<1
                assert(step_prob < 1)
                P[i, index] = step_prob

        corpus_prob = self.get_corpus_prob(inputted)
        max_prob = 0
        best_candidate = ''
        probs = []
        for i in range(len(self.corpus)):
            candidate = self.corpus[i]
            prob = corpus_prob[i]
            if len(candidate) == len(features):
                for i in range(len(features)):
                    letter = candidate[i]
                    index = ord(letter) - ord('a')

                    #[xc, yc] = self.letter_positions[index]
                    #[x, y] = positions[i]
                    #if (x - xc) ** 2 + (y - yc) ** 2 >= 2.1 ** 2:
                    #    prob = 0
                    #    break

                    step_prob = P[i, index]
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
            users = [1,2,3,4,5,6,8,9,10,12]#range(1, 13)
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
                folder_path = 'data-study1/' + str(user) + '-' + str(session) + '/'
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
                    pred, rank = self.predict(word_data, task[:end], word)
                    ranks.append(rank)
                    if pred != word:
                        # print('[Fail Cases]', word, pred)
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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('[Usage] python tool_analysis.py folder_name(x-x)')
        exit()

    # General
    Simulation().run()
    exit()

    # Personaliztion
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
