import cv2
import numpy as np


def mask_color(original_image):
    # HSV color
    boundaries = [
       ([-1], [0, 120, 120], [10, 255, 255])

       # ([1], [110, 50, 50], [130, 255, 255])

    ]

    original_img_hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

    for (code, lower, upper) in boundaries:
        lower = np.array(lower, dtype='uint8')
        upper = np.array(upper, dtype='uint8')

        # 첫번째 필터링
        mask = cv2.inRange(original_img_hsv, lower, upper)
        ret, thr = cv2.threshold(mask, 127, 255, 0)
        _, contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        copy_original_image = original_image.copy()

        trans_image_1 = cv2.drawContours(copy_original_image, contours, -1 ,(0,0,255),20)

        cv2.imshow('First',trans_image_1)
        cv2.waitKey(1)

        # 두번째 필터링 (마스크 씌우고, 붉은색 원 그린거에 다시 붉은색 검출하기)
        trans_img_1_hsv = cv2.cvtColor(trans_image_1, cv2.COLOR_BGR2HSV)
        mask2 = cv2.inRange(trans_img_1_hsv, lower, upper)

        cv2.imshow('Second',mask2)
        cv2.waitKey(1)

        # 잡음 제거
        kernel = np.ones((5,5))
        erosion = cv2.erode(mask2, kernel, iterations=4)
        delicate = cv2.dilate(erosion, kernel, iterations=5)

        cv2.imshow('Delete Noise',delicate)
        cv2.waitKey(1)


        ret, thr = cv2.threshold(delicate, 127, 255,0)
        _, contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(original_image, contours, -1, (0, 0, 255), 20)

        for c in contours:
            # get the bounding rect
            x, y, w, h = cv2.boundingRect(c)
            # draw a green rectangle to visualize the bounding rect
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # get the min area rect
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            # convert all coordinates floating point values to int
            box = np.int0(box)
            # draw a red 'nghien' rectangle
            #img = cv2.drawContours(original_image, [box], 0, (0, 0, 255))

            # convert all values to int
            center = (int(x), int(y))
            # and draw the circle in blue


            print(len(contours))
            print("center", center)


    return original_image


cap = cv2.VideoCapture('/Users/itaegyeong/Desktop/test3.mp4')

while True:
        ret, frame = cap.read()

        if ret == True:
            cv2.imshow('original', frame)
            cv2.imshow('result', mask_color(frame))

            cv2.waitKey(10)

