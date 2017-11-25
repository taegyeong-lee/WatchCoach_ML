import cv2
import numpy as np
import team_identification as ti
import math
import canvas_show as cs

# opencv 인식률 높히기 위해 테스트 진행 중
'''
붉은색 팀을 구별하기 위해서 사용하는 방법

A. 텐서플로우를 사용한 방법 (현재 방법)
사람 인식(텐서플로우) -> 예외처리(비정상적인 크기) -> 색상으로 팀 구별 -> 좌표생성

1. 문제점
- 매 이미지마다 인식을 해 속도느림
- 인식을 못하는 사람 발생
- 겹쳤을때, 갑자기 사라질때 탐지 불가능
- 오탐 발생

2. 해결방안
- 방법 B와 연동, 속도는 더 줄어듦, 정확도 향샹 보장 X

--------------------------------------------

B. opencv 를 사용한 방법 (현재 진행중)
붉은색 검출 -> 잡음제거 -> 붉은색 강조 -> 붉은색 검출 -> 좌표생성

1. 문제점
- 색이 정확하지 않을 경우 탐지 불가
- 색이 겹치거나 할 경우 오탐 발생

2. 해결방안
- 배경제거 -> 움직임 감지 및 테두리 -> 유니폼색 구별 -> 좌표반환
- 겹쳤을때? 갑자기 사라졌을때? 인식 자체가 안될때 ?
'''


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

            if w < 5 or h < 5 :
                continue

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow("maskClose", mask_close)
        cv2.imshow("maskOpen", mask_open)
        cv2.imshow("mask", mask)
        cv2.imshow("maskFinal",mask_final)
        cv2.imshow("frame2", frame2)
        cv2.imshow("frame",frame)

        out.write(frame)

        cv2.waitKey(1)



cam = cv2.VideoCapture('../video/test_soccer.mov')
out = cv2.VideoWriter('output.mov', -1, 20.0, (640, 360))

while True:
    ret, img = cam.read()
    mask_color(img)


