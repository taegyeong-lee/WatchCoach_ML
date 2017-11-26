import cv2
import numpy as np
import math


# '../video/test2_640x360.mov
# /Users/itaegyeong/Desktop/testshape.mov

video = cv2.VideoCapture('/Users/itaegyeong/Desktop/test_1.mov')
mog = cv2.createBackgroundSubtractorMOG2()

# 축소 팽창시키기
def morphology(frame):
    # Kernel Filter
    kernel_open = np.ones((1, 1))
    kernel_close = np.ones((20, 20))
    # morphology
    mask_open = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel_open, iterations=1)  # 축소
    mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel_close, iterations=1)  # 팽창
    return mask_close


# 움직이는 물체 탐지해서 이미지로 바꿔서 반환
def moving_object(frame):
    fg_mask = mog.apply(frame)
    fg_mask2 = cv2.medianBlur(fg_mask, 5)
    final_img = cv2.bitwise_and(frame, frame, mask=fg_mask2)
    return final_img


# 색상 필터링
def color_detection(frame, rgb_lower, rgb_upper, teamcode):
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array(rgb_lower, dtype='uint8')
    upper = np.array(rgb_upper, dtype='uint8')

    # Red mask
    mask = cv2.inRange(frame_hsv, lower, upper)
    return mask


# contours 구별
def contours_division(frame, mask):
    copy = frame.copy()
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in contours:
        x, y, w, h = cv2.boundingRect(i)
        #if w > h or w < 10 or len(i) < 40 or w > 70:
        #    continue

        cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return copy


# contours 구별
def contours_alg(frame, mask):
    copy = frame.copy()
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    point_list = []

    for i in contours:
        x, y, w, h = cv2.boundingRect(i)
        if w > h or w < 10 or len(i) < 40 or w > 70:
            continue
        cv2.drawContours(copy, i, -1, (0, 0, 255), 3)
        cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 0, 255), -1)

        point_list.append(i)

    #for i in range(0,len(point_list)):
    #    x1, y1, w1, h1 = cv2.boundingRect(point_list[i])
    #    for j in range(i + 1,len(point_list)):
    #        x2, y2, w2, h2 = cv2.boundingRect(point_list[j])
    #        distance = math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
    #        if distance < 20:
    #            cv2.line(copy,(x1,y1),(x2,y2),(0,255,0),3)

    return copy


def kmeans(frame, K=5):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)


all_list = []
frame_count = 0

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output2.mp4',fourcc, 20.0, (640,360))


while True:

    ret, frame = video.read()

    moving_frame = moving_object(frame)
    color_mask = color_detection(moving_frame, [0, 120, 80], [10, 255, 255], 1)

    contours_frame = contours_alg(frame,color_mask)


    color_mask2 = color_detection(contours_frame, [0, 244, 244], [2, 255, 255], 1)
    contours_frame2 = contours_division(frame, color_mask2)

    out.write(contours_frame2)


    #kmeans_frame = kmeans(frame,3)


    frame_count += frame_count
    cv2.imshow('contours_frame2', contours_frame2)
    cv2.imshow('contours', contours_frame)
    cv2.imshow('original', frame)
    cv2.imshow('moving_frame',moving_frame)
    cv2.imshow('color_frame', color_mask)
    cv2.imshow('color_frame2', color_mask2)

    #cv2.imshow('kmeans', kmeans_frame)
    cv2.waitKey(1)
