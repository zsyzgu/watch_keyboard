import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from astroML.stats import fit_bivariate_normal
from astroML.stats.random import bivariate_normal

def get_position(data): # get position from inputted data
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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('[Usage] python tool_analysis.py folder_name')
        exit()
    
    folder_path = 'data/' + sys.argv[1] + '/'
    N = 20
    points = [[] for ch in range(26)]
    for i in range(N):
        [task, inputted, data] = pickle.load(open(folder_path + str(i) + '.pickle', 'rb'))
        M = len(task)
        assert(len(inputted) == M and len(data) == M)

        for j in range(M):
            if task[j].isalpha():
                [side, which_finger, highlight_row, highlight_col, timestamp, palm_line, endpoint_x, endpoint_y, frame_id] = data[j]
                pos = get_position(data[j])
                ch = ord(task[j]) - ord('a')
                #points[ch].append(pos)
                if side == 'R':
                    endpoint_x += 20
                points[ch].append([endpoint_x, palm_line])

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

