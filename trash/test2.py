import cv2
import numpy as np

lowerBound = np.array([0, 70, 50])
upperBound = np.array([10, 255, 255])


cam = cv2.VideoCapture('/Users/itaegyeong/Desktop/tt.mov')

kernelOpen = np.ones((1, 1))
kernelClose = np.ones((20, 20))

while True:
    ret, img = cam.read()

    # convert BGR to HSV
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # create the Mask
    mask = cv2.inRange(imgHSV, lowerBound, upperBound)

    # morphology
    maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen,iterations=3)
    maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose,iterations=3)

    maskFinal = maskClose
    _, contours, _ = cv2.findContours(maskFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # cv2.drawContours(img, contours, -1, (255, 0, 0), 3)

    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)


    cv2.imshow("maskClose", maskClose)
    cv2.imshow("maskOpen", maskOpen)
    cv2.imshow("mask", mask)
    cv2.imshow("cam", img)
    cv2.waitKey(1)