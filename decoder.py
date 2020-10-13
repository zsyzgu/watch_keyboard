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

class Decoder:
    def __init__(self, is_tgrams = True):
        self.is_tgrams = is_tgrams
        self.init_touch_model()
        self.init_language_model()
    
    def init_touch_model(self):
        [self.positions, self.fingers, self.distributions] = pickle.load(open('touch.model', 'rb'))

    def init_language_model(self):
        self.size = 20000
        self.corpus = []
        self.corpus_map = {}
        self.corpus_prob = []
        lines = open('corpus.txt').readlines()
        for i in range(self.size):
            tags = lines[i].split(' ')
            word = tags[0]
            prob = float(tags[1])
            self.corpus.append(word)
            self.corpus_prob.append(prob)
            self.corpus_map[word] = i

        if self.is_tgrams:
            [self.bgrams_index, self.bgrams_freq] = pickle.load(open('2grams.model', 'rb'))
            [self.tgrams_index, self.tgrams_freq] = pickle.load(open('3grams.model', 'rb'))

    def get_finger(data):
        [side, finger, highlight_row, highlight_col, timestamp, palm_line, endpoint_x, endpoint_y] = data[:8]
        assert(side == 'L' or side == 'R')
        if side == 'L':
            return finger
        if side == 'R':
            return finger + 5

    def get_feature(data): # get position from inputted data
        [side, finger, highlight_row, highlight_col, timestamp, palm_line, endpoint_x, endpoint_y] = data[:8]
        if side == 'R':
            endpoint_x += 10
        return [endpoint_x, highlight_row]
        #return [endpoint_x, endpoint_y]

    def get_position(data):
        [side, finger, highlight_row, highlight_col, timestamp, palm_line, endpoint_x, endpoint_y] = data[:8]
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
        if self.is_tgrams:
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

    def predict(self, data, inputted, truth = None):
        features = [Decoder.get_feature(tap) for tap in data]
        fingers = [Decoder.get_finger(tap) for tap in data]
        positions = [Decoder.get_position(tap) for tap in data]

        P = np.zeros((len(features), 26)) # P(alpha | w_i)
        for i in range(len(features)):
            for alpha in range(26):
                finger = fingers[i]
                [x, y] = features[i]
                [xc, yc, std_x2, std_y2, std_xy, p] = self.distributions[alpha][finger]
                dx = x - xc
                dy = y - yc
                z = (dx ** 2) / std_x2 - (2 * p * dx * dy) / std_xy + (dy ** 2) / std_y2
                step_prob = self.fingers[alpha][finger]
                step_prob *= (.001 / (std_xy * ((1 - p ** 2) ** 0.5))) * math.exp(-z / (2 * (1 - p ** 2))) # the constant is modified to be small (1/2pi --> .01) so that prob<1
                assert(step_prob < 1)
                P[i, alpha] = step_prob

        corpus_prob = self.get_corpus_prob(inputted)
        max_prob = 0
        best_candidate = ''
        probs = []
        truth_prob = 0
        for i in range(len(self.corpus)):
            candidate = self.corpus[i]
            prob = corpus_prob[i]
            if len(candidate) == len(features):
                for i in range(len(features)):
                    letter = candidate[i]
                    alpha = ord(letter) - ord('a')

                    [xc, yc] = self.positions[alpha]
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
                
                if truth != None:
                    probs.append(prob)
                    if candidate == truth:
                        truth_prob = prob
        
        if truth == None:
            return best_candidate

        if truth_prob == 0:
            rank = -1
        else:
            rank = 1 + sum(np.array(probs) > truth_prob)
        return best_candidate, rank
