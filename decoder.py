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
    def __init__(self):
        self.init_touch_model()
        self.init_language_model()
    
    def init_touch_model(self):
        [self.positions, self.fingers, self.distributions] = pickle.load(open('touch_model.pickle', 'rb'))

    def init_language_model(self):
        self.corpus = []
        lines = open('corpus.txt').readlines()
        for i in range(20000):
            tags = lines[i].split(' ')
            word = tags[0]
            pri = float(tags[1])
            self.corpus.append([word, pri])

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
        return [endpoint_x, endpoint_y]

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

    def predict(self, data, truth = None):
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

        max_prob = 0
        best_candidate = ''
        probs = []
        truth_prob = 0
        for (candidate, prob) in self.corpus:
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
