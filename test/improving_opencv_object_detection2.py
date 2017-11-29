import cv2
import numpy as np
import math
import kmeans

#cv2.drawContours(copy, i, -1, (0, 0, 255), 3)
# '../input_sample_video/test2_640x360.mov
# /Users/itaegyeong/Desktop/testshape.mov

video = cv2.VideoCapture('/Users/itaegyeong/Desktop/20171127_195948.mov')
mog = cv2.createBackgroundSubtractorMOG2()

termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)


#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,360))

def cam_shift(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)
    ret, track_window = cv2.CamShift(dst, track_window, termination)


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


# 색상 강조
def contours_emphasis(frame, mask, o, fill):
    copy = frame.copy()
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    point_list = []

    for i in contours:
        x, y, w, h = cv2.boundingRect(i)
        if y < 130:
            continue

        cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 0, 255), fill)
        point_list.append(i)

    if o == 1:

        for i in range(0, len(point_list)):
            x1, y1, w1, h1 = cv2.boundingRect(point_list[i])

            if cv2.contourArea(point_list[i]) < 10:
                continue

            for j in range(i + 1, len(point_list)):
                x2, y2, w2, h2 = cv2.boundingRect(point_list[j])

                distance = math.sqrt(

                    ((x1 + w1 / 2) - (x2 + w2 / 2)) * ((x1 + w1 / 2) - (x2 + w2 / 2)) +
                    ((y1 + h1 / 2) - (y2 + h2 / 2)) * ((y1 + h1 / 2) - (y2 + h2 / 2))

                )

                if distance < 30 and cv2.contourArea(point_list[i]) < 100 and cv2.contourArea(point_list[j]) < 100:
                    cv2.line(copy, (x1, y1), (x2, y2), (0, 0, 255), 1)


    return copy


def kmeans(img):
    Z = img.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2



all_list = []
frame_count = 0

ret, frame = video.read()

# 움직이는 물체 감지
moving_frame = moving_object(frame)

while True:

    ret, frame = video.read()

    # 움직이는 물체 감지
    moving_frame = moving_object(frame)


    cv2.imshow('moving',frame)


    # 색상 강조 1번
    red_detection_mask = color_detection(moving_frame, [0, 120, 50], [30, 255, 255], 1)
   # red_detection_mask2 = color_detection(moving_frame, [150, 150, 50], [190, 255, 255], 1)

   # img = cv2.add(red_detection_mask,red_detection_mask2)

    # color_detection_mask = color_detection(moving_frame, [0, 120, 80], [10, 255, 255], 1)

    contours_emphasis_frame = contours_emphasis(moving_frame,red_detection_mask, 0, -1)

    cv2.imshow('contours_emphasis_frame',contours_emphasis_frame)


    # 색상 강조 2번
    color_detection_mask2 = color_detection(contours_emphasis_frame, [0, 120, 80], [10, 255, 255], 1)
    contours_emphasis_frame2 = contours_emphasis(contours_emphasis_frame,color_detection_mask2,1, -1)

    cv2.imshow('contours_emphasis_frame2',contours_emphasis_frame2)

    # 색상 강조 3번 및 출력
    color_detection_mask3 = color_detection(contours_emphasis_frame2, [0, 120, 80], [10, 255, 255], 1)
    contours_emphasis_frame3 = contours_emphasis(frame,color_detection_mask3, 1, 2)

    cv2.imshow('contours_emphasis_frame3',contours_emphasis_frame3)

    #red_detection_mask_shrink = cv2.resize(red_detection_mask, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    #red_detection_mask_shrink2 = cv2.resize(red_detection_mask2, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)



    color_detection_mask_shrink2 = cv2.resize(color_detection_mask2, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    color_detection_mask_shrink3 = cv2.resize(color_detection_mask3, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    contours_emphasis_frame_shrink = cv2.resize(contours_emphasis_frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    cv2.imshow('m', moving_frame)
   # cv2.imshow('color_detection_mask', red_detection_mask_shrink)
   # cv2.imshow('color_detection_mask2', red_detection_mask_shrink2)


   # cv2.imshow('frame', frame_shrink)


    cv2.waitKey(0)
