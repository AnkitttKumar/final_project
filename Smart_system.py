import cv2 as cv
import numpy as np
import face_recognition as face_rec
import os
# function to resize the images


def resize(img, size):
    width = int(img.shape[1] * size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv.resize(img, dimension, interpolation=cv.INTER_AREA)

path = 'Students'
studentImg = []
studentName = []
myList = os.listdir(path)

# print(myList)

# in order to store the name of the students we use for loop

for cl in myList:
    currImg = cv.imread(f'{path}\{cl}') ##student images /
    studentImg.append(currImg)
    studentName.append(os.path.splitext(cl)[0])

## print(studentName)

def findEncoding(images):
    encoding_list = []
    for img in images :
        img = resize(img, 0.5)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        encode_image = face_rec.face_encodings(img)[0]
        encoding_list.append(encode_image)
        return encoding_list


encode_List = findEncoding(studentImg)

vid = cv.VideoCapture(0)


while True:
    success, frame = vid.read()
    frame = cv.resize(frame, (0, 0), None, 0.25, 0.25)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)


    face_in_frame = face_rec.face_locations(frame)
    encode_in_frame = face_rec.face_encodings(frame, face_in_frame)
    


