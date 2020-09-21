import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from astroML.stats import fit_bivariate_normal
from astroML.stats.random import bivariate_normal
import math

class Illustration:
    def get_position(self, data): # get position from inputted data
        [side, index, highlight_row, highlight_col] = data[:4]
        #row = max(0-0.5,min(2+0.5,highlight_row - 1))
        #col = max(0-0.5,min(1+0.5,highlight_col - 1))
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
        return [col, row]

    def run(self):
        folder_path = 'data/' + sys.argv[1] + '/'
        N = 20
        points = [[] for ch in range(26)]
        for i in range(N):
            [task, inputted, data] = pickle.load(open(folder_path + str(i) + '.pickle', 'rb'))
            M = len(task)
            assert(len(inputted) == M and len(data) == M)

            for j in range(M):
                if task[j].isalpha():
                    ch = ord(task[j]) - ord('a')
                    #[side, which_finger, highlight_row, highlight_col, timestamp, palm_line, endpoint_x, endpoint_y, frame_id] = data[j]
                    pos = self.get_position(data[j])
                    points[ch].append(pos)

        fig, ax = plt.subplots()
        for ch in range(26):
            if len(points[ch]) > 0:
                X = np.array(points[ch])[:,0]
                Y = np.array(points[ch])[:,1]

                n_std = 3
                (center, a, b, theta) = fit_bivariate_normal(X, Y, robust=True)
                xc, yc = center
                ell = Ellipse(center, a * n_std, b * n_std, (theta * 180. / np.pi), ec='k', fc='none', color='red')
                plt.scatter(X, Y, color=('C'+str(ch)), s = 5)
                plt.scatter(xc, yc, color='red', s = 10)
                ax.add_patch(ell)
            else:
                print('lack of', chr(ch + ord('a')))
        plt.show()

class Simulation:
    def __init__(self):
        self.corpus = []
        lines = open('corpus.txt').readlines()
        for i in range(20000):
            tags = lines[i].split(' ')
            word = tags[0]
            pri = int(tags[1])
            self.corpus.append([word, pri])
        
        QWERTY = ['QWERTYUIOP', 'ASDFGHJKL', 'ZXCVBNM']
        self.letter_positions = [[-1, -1] for i in range(26)]
        for r in range(3):
            line = QWERTY[r]
            for c in range(len(line)):
                ch = line[c]
                index = ord(ch) - ord('A')
                self.letter_positions[index] = [c, r]
    
    def get_position(self, data): # get position from inputted data
        [side, index, highlight_row, highlight_col] = data[:4]
        #row = max(0-0.5,min(2+0.5,highlight_row - 1))
        #col = max(0-0.5,min(1+0.5,highlight_col - 1))
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
        return [col, row]

    def predict(self, positions, truth):
        N = len(self.corpus)
        max_pri = 0
        best_candidate = ''
        for i in range(N):
            (candidate, pri) = self.corpus[i]
            if len(candidate) == len(positions):
                for j in range(len(positions)):
                    letter = candidate[j]
                    index = ord(letter) - ord('a')
                    position = self.letter_positions[index]
                    dx = (position[0] - positions[j][0]) / 1
                    dy = (position[1] - positions[j][1]) / 1
                    pri *= math.exp(-((2*dx)**2+(2*dy)**2))
                if pri > max_pri:
                    max_pri = pri
                    best_candidate = candidate
                if candidate == truth:
                    truth_pri = pri
        
        score = truth_pri / max_pri
        if score < 1:
            score = 0
            #print(best_candidate, truth)
        return best_candidate, score

    def run(self):
        folder_path = 'data/' + sys.argv[1] + '/'
        N = 20
        scores = []
        for i in range(N):
            [task, inputted, data] = pickle.load(open(folder_path + str(i) + '.pickle', 'rb'))
            M = len(task)
            assert(len(inputted) == M and len(data) == M)

            if inputted != task:
                print('Error')
                continue

            words = []
            points = []
            truth = ''
            for j in range(M + 1):
                if j < M and task[j].isalpha():
                    pos = self.get_position(data[j])
                    points.append(pos)
                    truth += task[j]
                else:
                    pred, score = self.predict(points, truth)
                    words.append(pred)
                    scores.append(score)
                    points = []
                    truth = ''
        print(np.mean(scores), len(scores))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('[Usage] python tool_analysis.py folder_name')
        exit()
    #Illustration().run()
    Simulation().run()
