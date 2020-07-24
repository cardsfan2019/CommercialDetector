import numpy as np
from numpy import ones,vstack
from numpy.linalg import lstsq
import pyvjoy
import random
import cv2
import os
import time
from winKeyboard import Keyboard
from desktopmagic.screengrab_win32 import (
getDisplayRects, saveScreenToBmp, saveRectToBmp, getScreenAsImage,
getRectAsImage, getDisplaysAsImages)
import tensorflow as tf
from sound import Sound

for i in list(range(3))[::-1]:
    print(i + 1)
    time.sleep(1)

TOP = 200
LEFT = 2560
RIGHT = 2560 + 1920
BOTTOM = 1180
current_volume = 0
muted = False
model = tf.keras.models.load_model('baseballCommercialDetection2.model')
while True:
    screen = np.array(getRectAsImage((LEFT, TOP, RIGHT, BOTTOM)))
    screen = cv2.resize(screen, (100, 50))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2GRAY)
    result = list(model.predict([screen.reshape(-1, 100, 50, 1)])[0])
    if result[0] < .5:
        if muted == True:
            time.sleep(2)
            screenCheck = np.array(getRectAsImage((LEFT, TOP, RIGHT, BOTTOM)))
            screenCheck = cv2.resize(screenCheck, (100, 50))
            screenCheck = cv2.cvtColor(screenCheck, cv2.COLOR_BGRA2GRAY)
            check = list(model.predict([screenCheck.reshape(-1, 100, 50, 1)])[0])
            if check[0] < .5:
                Keyboard.key(Keyboard.VK_VOLUME_MUTE)
                muted = False
                print("Not a commercial")
    else:
        if muted == False:
            time.sleep(2)
            screenCheck = np.array(getRectAsImage((LEFT, TOP, RIGHT, BOTTOM)))
            screenCheck = cv2.resize(screenCheck, (100, 50))
            screenCheck = cv2.cvtColor(screenCheck, cv2.COLOR_BGRA2GRAY)
            check = list(model.predict([screenCheck.reshape(-1, 100, 50, 1)])[0])
            if check[0] > .5:
                print("Commercial!")
                Keyboard.key(Keyboard.VK_VOLUME_MUTE)
                muted = True
