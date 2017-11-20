import csv
import cv2
from sklearn.utils import shuffle

samples = []
with open('./data/dataset_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for sample in reader:
        samples.append(sample)

shuffle(samples)
current_path = './data/IMG/' + samples[0][0]
image = cv2.imread(current_path)

# crop to 40x320x3
new_img = image[70:135,:,:]

#save cropped image
cv2.imwrite('.\\documentation\\cropped_image.png',new_img)

cv2.imwrite('.\\documentation\\image.png',image)