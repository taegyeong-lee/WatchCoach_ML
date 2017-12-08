import cv2
import numpy as np


video = cv2.VideoCapture('/Users/itaegyeong/Desktop/dddd.mov')
knn = cv2.createBackgroundSubtractorKNN()

point = []

kernel = np.ones((3, 3), np.uint8)


while True:
    ret, frame = video.read()
    cv2.imshow('frame',frame)

    moving_mask = knn.apply(frame)

    ret, thresh1 = cv2.threshold(moving_mask, 127, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(moving_mask, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(moving_mask, 127, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(moving_mask, 127, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(moving_mask, 127, 255, cv2.THRESH_TOZERO_INV)

    cv2.imshow('thresh1', thresh1)
    cv2.imshow('thresh2', thresh2)
    cv2.imshow('thresh3', thresh3)
    cv2.imshow('thresh4', thresh4)
    cv2.imshow('thresh5', thresh5)

    opening2 = cv2.morphologyEx(thresh4, cv2.MORPH_OPEN, kernel, iterations=3)
    cv2.imshow('opening2 mask',opening2)




    _, contours, _ = cv2.findContours(thresh4, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    point_list = []
    copy = frame.copy()

    for i in contours:
        x, y, w, h = cv2.boundingRect(i)

        area = w*h

        if area < 500 or w>h:
            continue

        print(area)

        cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 0, 255), 2)


    cv2.imshow('c',copy)

    cv2.imshow('moving_mask',moving_mask)

    cv2.waitKey(0)
