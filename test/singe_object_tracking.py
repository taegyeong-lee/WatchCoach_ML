import cv2
import numpy as np

video = cv2.VideoCapture('/Users/itaegyeong/Desktop/testshape.mov')
mog = cv2.createBackgroundSubtractorMOG2()
termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)


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

    mask = cv2.inRange(frame_hsv, lower, upper)
    return mask



while True:
    ret, frame = video.read()

    # 움직이는 물체 감지
    moving_frame = moving_object(frame)

    # 색상 강조 1번
    red_detection_mask = color_detection(moving_frame, [0, 120, 50], [30, 255, 255], 1)

    cv2.imshow('red detection mask',red_detection_mask)
    cv2.imshow('red detection mask', red_detection_mask)


    cv2.waitKey(0)

