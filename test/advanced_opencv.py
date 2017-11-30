import cv2
import numpy as np


video = cv2.VideoCapture('/Users/itaegyeong/Desktop/3.mov')
mog = cv2.createBackgroundSubtractorMOG2()

termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)


knn = cv2.createBackgroundSubtractorKNN()


# 움직이는 물체 탐지해서 이미지로 바꿔서 반환
def moving_mask(frame):
    fg_mask = knn.apply(frame)


    cv2.imshow('fg', fg_mask)

    fg_mask2 = cv2.medianBlur(fg_mask, 5)
    final_img = cv2.bitwise_and(frame, frame, mask=fg_mask2)
    return final_img


# 색상 필터링
def color_detection(frame, rgb_lower, rgb_upper, teamcode):
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array(rgb_lower, dtype='uint8')
    upper = np.array(rgb_upper, dtype='uint8')

    mask = cv2.inRange(frame_hsv, lower, upper)
    return mask


# 색상 필터링
def color_detection2(frame, rgb_lower, rgb_upper, teamcode):
    #frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array(rgb_lower, dtype='uint8')
    upper = np.array(rgb_upper, dtype='uint8')

    mask = cv2.inRange(frame, lower, upper)
    return mask



# 색상 강조
def contours_emphasis(frame, mask, fill):
    copy = frame.copy()
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in contours:
        x, y, w, h = cv2.boundingRect(i)

        if w<10 or h <10 or h< 60:
            continue
        cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 0, 255), fill)

    return copy


while True:

    ret, frame = video.read()



    point = []

    #2진화 진화
    ret, thresh1 = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(frame, 127, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(frame, 127, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(frame, 127, 255, cv2.THRESH_TOZERO_INV)


    cv2.imshow('thresh1', thresh1)
    cv2.imshow('thresh2', thresh2)
    cv2.imshow('thresh3', thresh3)
    cv2.imshow('thresh4', thresh4)
    cv2.imshow('thresh5', thresh5)

    thresh1 = cv2.cvtColor(thresh1,cv2.COLOR_BGR2GRAY)
    th1 = cv2.adaptiveThreshold(thresh1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                cv2.THRESH_BINARY, 11, 2)
    th2 = cv2.adaptiveThreshold(thresh1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                cv2.THRESH_BINARY, 11, 2)

    cv2.imshow('th1',th1)
    cv2.imshow('th2',th2)


    # 움직이는 물체 감지
    moving_frame = moving_mask(frame)

    # 붉은색 물체 감지
    red_detection_mask = color_detection(moving_frame, [150, 150, 50], [190, 255, 255], 1)

    cv2.imshow('ff', red_detection_mask)
    cv2.imshow('moving', moving_frame)

    # 푸른색 물체 감지
    blue_detection_mask = color_detection2(frame, [0,0,0], [100, 200, 50], 1)
    cv2.imshow('blue',blue_detection_mask)


    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(red_detection_mask, kernel, iterations=2)

    erosion = cv2.erode(dilation, kernel, iterations=3)
    cv2.imshow("erosion", erosion)

    _, contours, _ = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        area = w * h
        if area < 200 or area > 2000 or w > h or y < 65:
            continue

        point.append([y, h, x, w])
        print(x, y, area)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


    cv2.imshow('frame',frame)

    cv2.waitKey(0)
