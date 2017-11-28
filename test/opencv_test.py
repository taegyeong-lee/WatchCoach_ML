import cv2
import numpy as np

video = cv2.VideoCapture('/Users/itaegyeong/Desktop/20171127_195948.mov')
mog = cv2.createBackgroundSubtractorMOG2()
termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)


# 축소 팽창시키기
def morphology(frame):
    # Kernel Filter
    kernel_open = np.ones((3, 3))
    kernel_close = np.ones((3, 3))
    # morphology
    mask_open = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel_open, iterations=1)  # 축소
    mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel_close, iterations=3)  # 팽창
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


# 색상 강조
def contours_emphasis(frame, mask, fill):
    copy = frame.copy()
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in contours:
        x, y, w, h = cv2.boundingRect(i)

        if w<10 or h <10:
            continue

        cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 0, 255), fill)

    return copy


while True:
    ret, frame = video.read()

    # 움직이는 물체 감지
    moving_mask = moving_object(frame)

    # 붉은색 물체 감지
    red_detection_mask = color_detection(frame, [150, 150, 50], [190, 255, 255], 1)
    contours_emphasis_frame = contours_emphasis(frame, red_detection_mask, -1)

    red_detection_mask2 = color_detection(contours_emphasis_frame, [0, 254, 254], [2, 255, 255], 1)
    contours_emphasis_frame = contours_emphasis(frame, red_detection_mask2, 2)



    cv2.imshow('red_mask1', red_detection_mask)
    cv2.imshow('contours_emphasis_mask', contours_emphasis_frame)


    cv2.waitKey(1)
