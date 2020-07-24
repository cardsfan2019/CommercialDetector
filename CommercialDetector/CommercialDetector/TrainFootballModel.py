import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D, Cropping2D

WIDTH = 100
HEIGHT = 50
LR = .001
EPOCHS = 8

train_data = np.load("normalized_football_data.npy")

train = train_data[:-250]
test = train_data[-250:]

X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
test_y = [i[1] for i in test]

def create_model():
    model = Sequential()
    model.add(Conv2D(320, (5,5), strides=(2,2), activation="relu", input_shape=(WIDTH,HEIGHT,1) ))
    model.add(Conv2D(160, (5,5), strides=(2,2), activation="relu" ))
    model.add(Conv2D(80, (5,5), strides=(2,2), activation="relu" ))
    model.add(Conv2D(40, (3,3), strides=(2,2), activation="relu" ))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(100))
    model.add(Dropout(0.3))
    model.add(Dense(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    # Compiling the network with mse loss function and the adam optimizer (No accuracy matrix because it's a regression problem)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create checkpoint callback
    model.fit([X], [Y], validation_data=([test_x], [test_y]), epochs=EPOCHS)

    return model

model = create_model()

model.save('commercialDetection.model')

