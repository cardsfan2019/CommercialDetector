import os
import keyboard
import time
import cv2
import numpy as np
from numpy import ones,vstack
from numpy.linalg import lstsq
from statistics import mean
from desktopmagic.screengrab_win32 import (
getDisplayRects, saveScreenToBmp, saveRectToBmp, getScreenAsImage,
getRectAsImage, getDisplaysAsImages)

file_name = 'baseball_training_data2.npy'

if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    training_data = list(np.load(file_name))
else:
    print('File does not exist, starting fresh!')
    training_data = []

for i in list(range(3))[::-1]:
    print(i + 1)
    time.sleep(1)

TOP = 400
LEFT = 2560
RIGHT = 2560 + 1920
BOTTOM = 1000

isCommercial = False
while True:
    screen = np.array(getRectAsImage((LEFT, TOP, RIGHT, BOTTOM)))
    screen = cv2.resize(screen, (100, 50))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2GRAY)
    #cv2.imshow('preview', screen)
    if keyboard.is_pressed('t'):
        isCommercial = True    
    if keyboard.is_pressed('f'):
        isCommercial = False
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
    
    training_data.append([screen, isCommercial])
    print(str(len(training_data)) + ", " + str(isCommercial))
    if len(training_data) % 500 == 0:
        np.save(file_name,training_data)

