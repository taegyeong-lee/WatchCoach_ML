import cv2
import numpy as np
import math
import kmeans


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
    point_list = []

    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

 #   for i in contours:
 #       x, y, w, h = cv2.boundingRect(i)
 #       if w > h or w < 10 or h < 3:
 #           continue

#        cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # 한번 더 예외처리
    for i in range(0,len(point_list)):
        x1, y1, w1, h1 = cv2.boundingRect(point_list[i])
        for j in range(i + 1,len(point_list)):
            x2, y2, w2, h2 = cv2.boundingRect(point_list[j])
            if (x1 < x2 and x1+w1 < x2+w2) or (x2 < x1 and x2+ w2 < x1 + w1):
                cv2.circle(copy, (int(x1), int(y1)),10, (0,0,255), -1)
                    #distance = math.sqrt(((x1+x1/2)-(x2+x2/2))*((x1+x1/2)-(x2+x2/2))+((y1+y1/2)-(y2+y2/2))*((y1+y1/2)-(y2+y2/2)))
            #if x2<x1:
            #    x1,x2 = x2,x1

            #if distance < 10 and h1 + h2 < 100:
            #    cv2.line(copy,(x1,y1),(x2,y2),(0,0,255),2)


    return copy


# contours 구별
'''
    # 한번 더 예외처리
    for i in range(0,len(point_list)):
        x1, y1, w1, h1 = cv2.boundingRect(point_list[i])
        for j in range(i + 1,len(point_list)):
            x2, y2, w2, h2 = cv2.boundingRect(point_list[j])
            if (x1 < x2 and x1+w1 < x2+w2) or (x2 < x1 and x2+ w2 < x1 + w1):
                cv2.circle(copy, (int(x1), int(y1)),10, (0,0,255), -1)
                    #distance = math.sqrt(((x1+x1/2)-(x2+x2/2))*((x1+x1/2)-(x2+x2/2))+((y1+y1/2)-(y2+y2/2))*((y1+y1/2)-(y2+y2/2)))
            #if x2<x1:
            #    x1,x2 = x2,x1

            #if distance < 10 and h1 + h2 < 100:
            #    cv2.line(copy,(x1,y1),(x2,y2),(0,0,255),2)
            '''

#cv2.drawContours(copy, i, -1, (0, 0, 255), 3)
def contours_emphasis(frame, mask):

    copy = frame.copy()
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 예외처리 후 리스트에 좌표점들 추가
    point_list = []
    for i in contours:
        x, y, w, h = cv2.boundingRect(i)

        if y < 130:
            continue
        cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 0, 255), -1)

    return copy


def kmeans(img):
    Z = img.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 5
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2



all_list = []
frame_count = 0

while True:

    ret, frame = video.read()

    # 움직이는 물체 감지
    moving_frame = moving_object(frame)
    # 색상 감지 1번
    color_detection_mask = color_detection(moving_frame, [0, 120, 80], [10, 255, 255], 1)
    # 색상 강조 1번
    contours_emphasis_frame = contours_emphasis(frame,color_detection_mask)

    # ---------------------

    # 색상 감지 2번
    color_detection_mask2 = color_detection(contours_emphasis_frame, [0, 120, 80], [10, 255, 255], 1)

    # 색상 강조 2번
    contours_emphasis_frame2 = contours_emphasis(contours_emphasis_frame,color_detection_mask2)

    # 강조한 색상 감지
    color_emphasis_mask = color_detection(contours_emphasis_frame2, [0, 244, 244], [2, 255, 255], 1)

    # 물체 인식
    result_frame = contours_division(frame, color_emphasis_mask)

    frame_count += frame_count

    cv2.imshow('color_detection_mask', color_detection_mask)
    cv2.imshow('color_emphasis_mask', color_emphasis_mask)
    cv2.imshow('contours_emphasis_frame2', color_detection_mask2)
    cv2.imshow('result_frame', result_frame)

    cv2.waitKey(0)
