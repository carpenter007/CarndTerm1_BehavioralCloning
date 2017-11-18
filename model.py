import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Read in lines of driving log file
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for sample in reader:
        samples.append(sample)

# augmentation was done previously
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print(train_samples[0])
print(validation_samples[0])

# Load batches of images and angles and store them into X_train and y_train 
def generator(stamp, batch_size=32):
    num_samples = len(stamp)
    while 1: # Loop forever so the generator never terminates
        shuffle(stamp)
        for offset in range(0, num_samples, batch_size):
            batch_samples = stamp[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                current_path = './data/IMG/' + sample[0].split('\\')[-1]
                image = cv2.imread(current_path)
                images.append(image)
                angle = float(sample[3])
                angles.append(angle)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Check the shapes
print("Number of Images:" + str(len(samples)))

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

#X_train, y_train = next(train_generator)
#print(X_train.shape)
#print(y_train.shape)

model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=4)

model.save('model.h5')
