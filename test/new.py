import cv2
import numpy as np
import matplotlib.pyplot as plt
import te


def histogram(img):
    hist = cv2.calcHist([img],[2],None,[256],[0,256])
    plt.plot(hist, color='r')
    plt.xlim([0,256])
    plt.show()


video = cv2.VideoCapture('/Users/itaegyeong/Desktop/good.mov')
knn = cv2.createBackgroundSubtractorKNN()
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('test_basketball_output.mp4',fourcc, 20.0, (640,360))

while True:
    ret, frame = video.read()
    cv2.imshow('frame',frame)

    # Morphology의 opening, closing을 통해서 노이즈나 Hole제거
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel, iterations=1)
    cv2.imshow('opening',opening)


    opening_hsv = cv2.cvtColor(opening,cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(opening_hsv, np.array((150., 150., 50.)), np.array((190., 255., 255.)))
    cv2.imshow('masks', mask)

    opening2 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cv2.imshow('opening2 mask',opening2)

    _, contours, _ = cv2.findContours(opening2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        area = w * h
        #if area < 200 or y < 40:
        #    continue


        print(x, y, area)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


    cv2.imshow('frame',frame)
    out.write(frame)



    cv2.waitKey(0)

