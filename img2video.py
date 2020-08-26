import cv2
import glob
import numpy as np


def resize(img_array, align_mode):
    _height = len(img_array[0])
    _width = len(img_array[0][0])
    for i in range(1, len(img_array)):
        img = img_array[i]
        height = len(img)
        width = len(img[0])
        if align_mode == 'smallest':
            if height < _height:
                _height = height
            if width < _width:
                _width = width
        else:
            if height > _height:
                _height = height
            if width > _width:
                _width = width
 
    for i in range(0, len(img_array)):
        img1 = cv2.resize(img_array[i], (_width, _height), interpolation=cv2.INTER_CUBIC)
    
    return img_array, (_width, _height)

def to_mp4(img_array):
    fps = 30
    img_array, size = resize(img_array, 'largest')
    out = cv2.VideoWriter('result.mp4', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
 
    N = len(img_array)
    for i in range(N):
        print(str(i) + '/' + str(N))
        out.write(img_array[i])
    out.release()

def run(start, end):
    path = "result/"
    img_array = []
 
    #for filename in glob.glob(path+'/*.png'):
    for id in range(start, end):
        filename = path + str(id) + '.png'
        img = cv2.imread(filename)
        if img is None:
            #print(filename + " is error!")
            continue
        img_array.append(img)
 
    to_mp4(img_array)
 
if __name__ == "__main__":
    run(100, 500)
