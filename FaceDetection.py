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


# Image declaration

ankit = face_rec.load_image_file('Samples\Sample.jpg')
ankit = cv.cvtColor(ankit, cv.COLOR_BGR2RGB)
ankit = resize(ankit, 0.50)
ankit_test = face_rec.load_image_file('Samples\Test.jpg')
ankit_test = cv.cvtColor(ankit_test, cv.COLOR_BGR2RGB)
ankit_test = resize(ankit_test, 0.50)


# Finding face location

face_location_ankit = face_rec.face_locations(ankit)[0]
encode_ankit = face_rec.face_encodings(ankit)[0]
cv.rectangle(ankit, (face_location_ankit[3], face_location_ankit[0]), (face_location_ankit[1], face_location_ankit[2]), (255, 0, 0), 3)


face_location_ankit_test = face_rec.face_locations(ankit_test)[0]
encode_ankit_test = face_rec.face_encodings(ankit_test)[0]
cv.rectangle(ankit_test, (face_location_ankit[3], face_location_ankit[0]), (face_location_ankit[1], face_location_ankit[2]), (255, 0, 0), 3)

results = face_rec.compare_faces([encode_ankit], encode_ankit_test)
print(results)

cv.putText(ankit_test, f'{results}', (50,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


cv.imshow('main_img', ankit)
cv.imshow('test_img', ankit_test)

cv.waitKey(0)
cv.destroyAllWindows()
