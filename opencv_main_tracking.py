import cv2
import numpy as np
import math
from scipy.spatial.distance import pdist





video = cv2.VideoCapture('/Users/itaegyeong/Desktop/good.mov')
knn = cv2.createBackgroundSubtractorKNN()


# 이전의 프레임과 비교하여 w,h 도 크게 차이나지 않으며, 거리도 가장 작은것으로 선택
# 각 선택된것은 선수 1,2,3,4,5 좌표값으로 저장 후 칼만필터를 이용해서 추적하게 처리

def distance(pre_frame, current_frame):

    min_num = []

    for pre in range(0, len(pre_frame)):

        min_dis = math.sqrt((pre_frame[pre][0] - current_frame[0][0]) * (pre_frame[pre][0] - current_frame[0][0])
                        + (pre_frame[pre][1] - current_frame[0][1]) * (pre_frame[pre][1] - current_frame[0][1]))


        for curr in range(1, len(current_frame)):

            dis = math.sqrt((pre_frame[pre][0] - current_frame[curr][0]) * (pre_frame[pre][0] - current_frame[curr][0])
                            + (pre_frame[pre][1] - current_frame[curr][1]) * (pre_frame[pre][1] - current_frame[curr][1]))

            if min_dis > dis:
                min_dis = dis
                min_num = [pre, curr]

        if min_num != []:
            print(min_num,pre_frame[min_num[0]],current_frame[min_num[1]])

    #print(min_num, pre_frame[min_num[0]], current_frame[min_num[1]])
    print("\n")




ret, frame = video.read()
point = []
frame_point = []
frame_count = 0

# Morphology의 opening, closing을 통해서 노이즈제거
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel, iterations=1)

# 붉은색 mask 씌우기
opening_hsv = cv2.cvtColor(opening,cv2.COLOR_BGR2HSV)
mask = cv2.inRange(opening_hsv, np.array((150., 150., 50.)), np.array((190., 255., 255.)))

# 붉은색 mask에 Morpholgy로 잡음 제거
opening2 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

_, contours, _ = cv2.findContours(opening2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
for c in contours:
    x, y, w, h = cv2.boundingRect(c)

    area = w * h
    if area < 200 or y < 40:
        continue

    point.append([x, w, y, h])
    # print(x, y, area)

frame_point.append(point)
frame_count = frame_count + 1



while True:
    point = []

    ret, frame = video.read()
    cv2.imshow('frame',frame)

    # Morphology의 opening, closing을 통해서 노이즈제거
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel, iterations=1)
    cv2.imshow('opening',opening)

    # 붉은색 mask 씌우기
    opening_hsv = cv2.cvtColor(opening,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(opening_hsv, np.array((150., 150., 50.)), np.array((190., 255., 255.)))
    cv2.imshow('masks', mask)

    # 붉은색 mask에 Morpholgy로 잡음 제거
    opening2 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cv2.imshow('opening2 mask',opening2)


    _, contours, _ = cv2.findContours(opening2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    a=0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        area = w * h
        if area < 200 or y < 40:
            continue
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        point.append([x,w,y,h])


    frame_point.append(point)


    distance(frame_point[frame_count-1],frame_point[frame_count])

    frame_count = frame_count + 1

    cv2.imshow('frame',frame)
    cv2.waitKey(0)

