import cv2
import numpy as np

video = cv2.VideoCapture('/Users/itaegyeong/Desktop/good.mov')
mog = cv2.createBackgroundSubtractorMOG2()
termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)



# 축소 팽창시키기
def morphology(frame):
    # Kernel Filter
    kernel_open = np.ones((3, 3))
    kernel_close = np.ones((3, 3))
    # morphology
    mask_open = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel_open, iterations=10)  # 축소
    # mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel_close, iterations=3)  # 팽창
    return mask_open


# 움직이는 물체 탐지해서 이미지로 바꿔서 반환
def moving_mask(frame):
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

        if w<10 or h <10 or h< 60:
            continue
        cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 0, 255), fill)

    return copy


ret, frame = video.read()

point = []

#first_frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)



# 움직이는 물체 감지
moving_frame = moving_mask(frame)

# 붉은색 물체 감지
red_detection_mask = color_detection(frame, [150, 150, 50], [190, 255, 255], 1)
kernel = np.ones((5, 5), np.uint8)
dilation = cv2.dilate(red_detection_mask, kernel, iterations=3)

erosion = cv2.erode(dilation, kernel, iterations=3)
cv2.imshow("erosion",erosion)


_, contours, _ = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
for c in contours:
    x, y, w, h = cv2.boundingRect(c)

    area = w * h
    if area < 200 or area > 1500 or w>h:
        continue

    point.append([y,h,x,w])
    print(x,y,area)


    # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


r, h, c, w = point[1][0], point[1][1], point[1][2], point[1][3]  # simply hardcoded the values

track_window = (c, r, w, h)
roi = frame[r:r + h, c:c + w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

mask = cv2.inRange(hsv_roi, np.array((150., 150., 50.)), np.array((190., 255., 255.)))

roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])

cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

cv2.imshow('f',frame)

while True:
    ret, frame = video.read()



    for p in point:

        r, h, c, w = p[0], p[1], p[2], p[3]  # simply hardcoded the values

        track_window = (c, r, w, h)
        roi = frame[r:r + h, c:c + w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv_roi, np.array((150., 150., 50.)), np.array((190., 255., 255.)))

        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])

        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        cv2.imshow('back',dst)

        ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        # Draw it on image
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img = cv2.polylines(frame, [pts], True, 255, 2)

        cv2.circle(img, (int(c+w/2),int(r+h/2)),3, [0,255,0])



       #x, y, w, h = track_window
       # img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)


        cv2.imshow('img',img)





    # 2진화 진화
    #ret, thresh1 = cv2.threshold(red_detection_mask, 127, 255, cv2.THRESH_BINARY)
    #ret, thresh2 = cv2.threshold(red_detection_mask, 127, 255, cv2.THRESH_BINARY_INV)
    # ret, thresh3 = cv2.threshold(red_detection_mask, 127, 255, cv2.THRESH_TRUNC)
    # ret, thresh4 = cv2.threshold(red_detection_mask, 127, 255, cv2.THRESH_TOZERO)
    # ret, thresh5 = cv2.threshold(red_detection_mask, 127, 255, cv2.THRESH_TOZERO_INV)

    # cv2.imshow('moving_frame', moving_frame)
    #
    # cv2.imshow('thresh1', thresh1)
    # cv2.imshow('thresh2', thresh2)
    # cv2.imshow('thresh3', thresh3)
    # cv2.imshow('thresh4', thresh4)
    # cv2.imshow('thresh5', thresh5)
    #
    #
    # th1 = cv2.adaptiveThreshold(thresh1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
    #                             cv2.THRESH_BINARY, 11, 2)
    # th2 = cv2.adaptiveThreshold(thresh1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
    #                             cv2.THRESH_BINARY, 11, 2)


    red_mask_th1 = cv2.adaptiveThreshold(red_detection_mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                cv2.THRESH_BINARY, 11, 2)


    _, contours, _ = cv2.findContours(red_mask_th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in contours:
        x, y, w, h = cv2.boundingRect(i)


    # red_mask_th2 = cv2.adaptiveThreshold(red_detection_mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
    #                            cv2.THRESH_BINARY, 11, 2)


    # kernel = np.ones((5, 5), np.uint8)
    # dilation = cv2.dilate(red_mask_th1, kernel, iterations=1)
    # cv2.imshow("dis", dilation)


    # erosion = cv2.erode(red_mask_th1, kernel, iterations=3)
    # cv2.imshow("erosion",erosion)




    # cv2.imshow('red_mask_th1', red_mask_th1)
   # cv2.imshow('red_mask_th2', red_mask_th2)
   #  cv2.imshow('red_mask1', red_detection_mask)
   #
   #  cv2.imshow('frame', frame)




    cv2.waitKey(0)
