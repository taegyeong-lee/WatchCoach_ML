import cv2
import numpy as np


# '../video/test2_640x360.mov
# /Users/itaegyeong/Desktop/testshape.mov

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


# 색상 추적 및 예외처리
def object_division(color_mask):
    _, contours, _ = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    center = []

    for i in contours:
        x, y, w, h = cv2.boundingRect(i)

        # 영상(상황)에 맞춰 커스텀하게 변경해야 함
        if w < 10 or h < 30 or w > 60 or h > 60 or y < 140:
            continue
        #if len(contours[i]) < 30 or cv2.contourArea(contours[i]) < 600:
        #    continue

        #cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #cv2.circle(copy,(int(x+w/2),int(y+h/2)),3,(0,255,0),2)

        center.append([int(x + w / 2), int(y + h / 2)])

    return center


def object_tracking(all_center_list, frame_count):

    # object count increasing
    # 어떻게 처리할것인가 ?, 트래킹을 할것인가 ?
    # 트래킹 한다면 정확도는 어느정도 되는가 ?, 실시간으로 인식할것인가 ?

    if len(all_center_list[frame_count]) < len(all_center_list[frame_count-1]):
        a=1



all_center_list = []
frame_count = 0

while True:
    ret, frame = video.read()

    moving_frame = moving_object(frame)
    color_mask = color_detection(moving_frame,[0, 120, 120], [10, 255, 255],1)

    center_list = object_division(color_mask)

    all_center_list.append(center_list)

    if frame_count != 0:
        object_list = object_tracking(all_center_list,frame_count)


    frame_count += frame_count


    cv2.imshow('original', frame)
    cv2.imshow('moving_frame',moving_frame)
    cv2.imshow('color_frame', color_mask)


    cv2.waitKey(1)

