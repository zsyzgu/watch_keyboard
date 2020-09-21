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
            pri = int(tags[1])
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
            data = kwargs['data']
            task = kwargs['task']
            assert(len(data) == len(task))
            points = [[] for ch in range(26)]
            for i in range(len(task)):
                letter = task[i]
                if letter.isalpha():
                    index = ord(letter) - ord('a')
                    position = self.get_position(data[i])
                    points[index].append(position)
            
            fig, ax = plt.subplots()
            ans = []
            for index in range(26):
                if len(points[index]) > 0:
                    X = np.array(points[index])[:,0]
                    Y = np.array(points[index])[:,1]

                    n_std = 1
                    (center, a, b, theta) = fit_bivariate_normal(X, Y, robust=False)
                    xc, yc = center
                    ell = Ellipse(center, a * n_std, b * n_std, (theta * 180. / np.pi), ec='k', fc='none', color='red')
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
        return [col, row]
        #return [endpoint_x, endpoint_y]
        #return [endpoint_x, palm_line]

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
                    pri *= ((2 * math.pi * std_xy * ((1 - p ** 2) ** 0.5)) ** -1) * math.exp(-z / (2 * (1 - p ** 2)))
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

    def run(self):
        folder_path = 'data/' + sys.argv[1] + '/'
        N = 20

        total_task = ''
        total_data = []
        for i in range(N):
            [task, inputted, data] = pickle.load(open(folder_path + str(i) + '.pickle', 'rb'))
            M = len(task)
            assert(len(inputted) == M and len(data) == M)
            total_task += task
            total_data.extend(data)
        self.calc_letter_distribution(mode=self.LETTER_DISTRIBUTION_STAT, data=total_data, task=total_task)

        ranks = []
        for i in range(N):
            [task, inputted, data] = pickle.load(open(folder_path + str(i) + '.pickle', 'rb'))

            if inputted != task:
                print('Error')
                continue

            words = []
            points = []
            truth = ''
            for j in range(len(task) + 1):
                if j < len(task) and task[j].isalpha():
                    pos = self.get_position(data[j])
                    points.append(pos)
                    truth += task[j]
                else:
                    pred, rank = self.predict(points, truth)
                    words.append(pred)
                    ranks.append(rank)
                    points = []
                    truth = ''

        print('=====   Top-5 accuracy   =====')
        ranks = np.array(ranks)
        for rank in range(1, 6):
            print(sum(ranks == rank) / len(ranks))
            

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('[Usage] python tool_analysis.py folder_name')
        exit()
    #Illustration().run()
    Simulation().run()
