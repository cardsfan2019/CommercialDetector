import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2

train_data = np.load('baseball_training_data2.npy')

df = pd.DataFrame(train_data)
print(df.head())
print(Counter(df[1].apply(str)))

commercial = []
notCommercial = []

for data in train_data:
    img = data[0]
    isCommercial = data[1]
    
    if isCommercial:
        commercial.append([img, True])
    else:
        notCommercial.append([img, False])

commercial = commercial[:len(notCommercial)]
notCommercial = notCommercial[:len(commercial)]
final_data = commercial + notCommercial

shuffle(final_data)
print(len(train_data))
print(len(final_data))
np.save('baseball_normalized_data2.npy', final_data)

df = pd.DataFrame(final_data)
print(df.head())
print(Counter(df[1].apply(str)))
