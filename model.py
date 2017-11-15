import csv
import cv2
import numpy as np

# Read in lines of driving log file
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Load images and measurements and store them into X_train and y_train 
correction = 0.3 # this is a parameter to tune the left and right correction for left and right camera
images = []
measurements = []
for line in lines:
    for i in range(1):
        source_path = line[i]
        filename = source_path.split('\\')[-1]
        current_path = './data/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        # create adjusted steering measurements for the side camera images
        measurement = float(line[3])
        if(i == 0):
            # nothing to do
            measurement = measurement
        elif(i == 1):
            # steering more to right (left camera picture)
            measurement = measurement + correction
        elif(i == 2):
            # steering more to left (right camera picture)
            measurement = measurement - correction
        else:
            print("Error, unexpected loop range")
        measurements.append(measurement)

augmented_images, augmented_measurements = [], []
for image,measurement in zip(images,measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(cv2.flip(image,1))
        augmented_measurements.append(measurement*-1.0)


# Check the shapes
print("Number of Images:" + str(len(images)))
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

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
print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
model.fit(X_train, y_train, nb_epoch=4, validation_split=0.2, shuffle=True)

model.save('model.h5')

