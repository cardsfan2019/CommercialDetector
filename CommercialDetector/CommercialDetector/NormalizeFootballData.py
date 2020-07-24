import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2

train_data = np.load('training_data.npy')

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
final_data = commercial + notCommercial

shuffle(final_data)
print(len(train_data))
print(len(final_data))
np.save('football_normalized_data.npy', final_data)

df = pd.DataFrame(final_data)
print(df.head())
print(Counter(df[1].apply(str)))
