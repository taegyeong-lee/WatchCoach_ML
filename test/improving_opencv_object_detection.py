import cv2
import numpy as np
import team_identification as ti
import math
import canvas_show as cs

# opencv 인식률 높히기 위해 테스트 진행 중

def mask_color(frame):

    # HSV color
    boundaries = [
        ([-1], [0, 120, 120], [10, 255, 255]),
        ([1], [18, 0, 0], [20, 255, 255])
        #([1], [110, 50, 50], [130, 255, 255])
    ]

    # Kernel Filter
    kernelOpen = np.ones((1, 1))
    kernelClose = np.ones((20, 20))

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for (code, lower, upper) in boundaries:

        lower = np.array(lower, dtype='uint8')
        upper = np.array(upper, dtype='uint8')

        # Red mask
        mask = cv2.inRange(frame_hsv, lower, upper)

        # morphology
        mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen, iterations=3) # 축소
        mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernelClose, iterations=2) # 팽창

        # Histogram
        mask_final = cv2.medianBlur(mask_close,5)

        _, contours, _ = cv2.findContours(mask_final.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Red mask 2
        frame2 = cv2.drawContours(frame.copy(), contours, -1, (0, 0, 255), 20)
        frame2_hsv = cv2.cvtColor(frame2.copy(),cv2.COLOR_BGR2HSV)
        mask2 = cv2.inRange(frame2_hsv, lower, upper)

        _, contours, _ = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        for i in range(len(contours)):

            x, y, w, h = cv2.boundingRect(contours[i])
            print(len(contours[i]),cv2.contourArea(contours[i]),x,w,y,h)

            if w < 30 or h < 30 or w > 70 or h > 70 or y < 70 :
                continue

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow("maskClose", mask_close)
        cv2.imshow("maskOpen", mask_open)
        cv2.imshow("mask", mask)
        cv2.imshow("maskFinal",mask_final)
        cv2.imshow("frame2", frame2)
        cv2.imshow("frame",frame)

        cv2.waitKey(1)




cam = cv2.VideoCapture('/Users/itaegyeong/Desktop/2.mp4')

while True:
    ret, img = cam.read()
    mask_color(img)


