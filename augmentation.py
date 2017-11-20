import csv
import cv2

# Read in lines of driving log file
print("create new files")
new_rows = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        new_rows.append(row)
        current_path = './data/IMG/' + row[0].split('\\')[-1]
        #Save flipped image
        image = cv2.imread(current_path)
        image = cv2.flip(image,1)
        cv2.imwrite('./data/IMG/flip_' + row[0].split('\\')[-1],image)

#print("append new images to driving_log")
#with open(r'./data/driving_log.csv', 'a',newline='') as csvfile:
#    writer = csv.writer(csvfile)
#    for row in new_rows:
#        row[0].split('\\')[-1] = 'flip_' + row[0].split('\\')[-1]
#        row[3] = str(float(row[3])*-1.0)
#        writer.writerow(row)

newRow = [0,0]
print("create new driving_log")
with open('./data/dataset_log.csv', 'w',newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in new_rows:
        newRow[0] = 'flip_' + row[0].split('\\')[-1]
        newRow[1] = str(float(row[3])*-1.0)
        #resultRows.append(newRow)
        writer.writerow(newRow)
        newRow[0] = row[0].split('\\')[-1]
        newRow[1] = str(float(row[3])*1.0)
        #resultRows.append(newRow)
        writer.writerow(newRow)
        newRow[0] = row[1].split('\\')[-1]
        newRow[1] = str(float(row[3])*1.0 + 0.12)
        #resultRows.append(newRow)
        writer.writerow(newRow)
        newRow[0] = row[2].split('\\')[-1]
        newRow[1] = str(float(row[3])*1.0 - 0.12)
        #resultRows.append(newRow)
        writer.writerow(newRow)
