import cv2
import numpy as np
import team_identification as ti
import math
import canvas_show as cs


def mask_color(frame):

    # HSV color
    boundaries = [
        ([-1], [0, 120, 120], [10, 255, 255]),
        #([1], [18, 0, 0], [20, 255, 255])
        #([1], [110, 50, 50], [130, 255, 255])
    ]

    # Kernel Filter
    kernel_open = np.ones((1, 1))
    kernel_close = np.ones((20, 20))


    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for (code, lower, upper) in boundaries:

        lower = np.array(lower, dtype='uint8')
        upper = np.array(upper, dtype='uint8')

        # Red mask
        mask = cv2.inRange(frame_hsv, lower, upper)

        # morphology
        mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=10) # 축소
        mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel_close, iterations=2) # 팽창

        # Histogram
        mask_final = cv2.medianBlur(mask_close,5)

        _, contours, _ = cv2.findContours(mask_final.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Red mask 2
        frame2 = cv2.drawContours(frame.copy(), contours, -1, (0, 0, 255), 10)
        frame2_hsv = cv2.cvtColor(frame2.copy(),cv2.COLOR_BGR2HSV)
        mask2 = cv2.inRange(frame2_hsv, lower, upper)

        _, contours, _ = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for i in range(len(contours)):

            x, y, w, h = cv2.boundingRect(contours[i])

            if w < 10 or h < 30 or w > 60 or h > 100 or y < 140:
                continue

            if len(contours[i]) < 30 or cv2.contourArea(contours[i]) < 600:
                continue

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow("maskClose", mask_close)
        cv2.imshow("maskOpen", mask_open)
        cv2.imshow("mask", mask)
        cv2.imshow("maskFinal",mask_final)
        cv2.imshow("frame2", frame2)
        cv2.imshow("frame",frame)

        cv2.waitKey(1)


# '../video/test2_640x360.mov
# /Users/itaegyeong/Desktop/testtest2.mov
video = cv2.VideoCapture('/Users/itaegyeong/Desktop/testshape.mov')

mog = cv2.createBackgroundSubtractorMOG2()


# 축소 팽창시키기
def morphology(frame):
    # Kernel Filter
    kernel_open = np.ones((1, 1))
    kernel_close = np.ones((20, 20))
    # morphology
    mask_open = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel_open, iterations=10)  # 축소
    mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel_close, iterations=2)  # 팽창
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


# 색상
def color_tracking(color_mask):
    _, contours, _ = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in contours:
        x, y, w, h = cv2.boundingRect(i)

        # if w < 10 or h < 30 or w > 60 or h > 100 or y < 140:
        #    continue

        # if len(contours[i]) < 30 or cv2.contourArea(contours[i]) < 600:
        #    continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)


while True:
    ret, frame = video.read()

    moving_frame = moving_object(frame)
    color_mask = color_detection(moving_frame,[0, 120, 120], [10, 255, 255],1)


    cv2.imshow('original', frame)
    cv2.imshow('moving_frame',moving_frame)
    cv2.imshow('color_frame', color_mask)




    cv2.waitKey(1)
    '''
        for i in range(len(contours)):

            x, y, w, h = cv2.boundingRect(contours[i])
            if cv2.contourArea(contours[i]) < 20:
                continue

            if h > 80 or w > 80 or w>h or y <130 or y > 329 or w*h < 200:
                continue


            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)

    '''
   # mask_color(frame)


