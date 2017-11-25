import cv2
import numpy as np
import team_identification as ti
import math

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

        copy_original_image = original_image.copy()
        copy_original_image2 = original_image.copy()


        # 첫번째 필터링, 붉은색 hsv 검출하기
        mask = cv2.inRange(original_img_hsv, lower, upper)
        ret, thr = cv2.threshold(mask, 127, 255, 0)
        _, contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 첫번째 필터 그리기, 붉은색 추가
        trans_image_1 = cv2.drawContours(copy_original_image, contours, -1 ,(0,0,255),10)

        cv2.imshow('First',trans_image_1)
        cv2.waitKey(1)

        # 두번째 필터링, 붉은색이 추가된 이미지에 다시 붉은색 검출하기
        trans_img_1_hsv = cv2.cvtColor(trans_image_1, cv2.COLOR_BGR2HSV)
        mask2 = cv2.inRange(trans_img_1_hsv, lower, upper)

        mask2 = cv2.medianBlur(mask2, 5)
        mask2 = cv2.bilateralFilter(mask2, 9, 75, 75)
        mask2 = cv2.GaussianBlur(mask2,(5,5),0)

        cv2.imshow('Second',mask2)
        cv2.waitKey(1)

        # 잡음 제거
        #kernel = np.ones((5,5))
        #erosion = cv2.erode(mask2, kernel, iterations=6)
        #delicate = cv2.dilate(erosion, kernel, iterations=5)


        ret, thr = cv2.threshold(mask2, 127, 255,0)
        _, contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(original_image, contours, -1, (0, 0, 255), 20)

        cv2.imshow('Delete mask',original_image)
        cv2.waitKey(1)

        boxes = []

        print(len(contours))


        for c in contours:
            # get the bounding rect


            x, y, w, h = cv2.boundingRect(c)

            # print(len(c))
            if len(c) < 20 or y < 160:
                continue


            center = (int(x), int(y))

            # draw a green rectangle to visualize the bounding rect
            cv2.rectangle(copy_original_image2, (x, y), (x + w, y + h), (0, 255, 0), 2)

            pre_center = (int(x), int(y))

            cv2.imshow('box', copy_original_image2)
            cv2.waitKey(1)



            # get the min area rect
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            # convert all coordinates floating point values to int
            box = np.int0(box)
            # draw a red 'nghien' rectangle
            #img = cv2.drawContours(original_image, [box], 0, (0, 0, 255))

            # convert all values to int
            center = (int(x), int(y))
            boxes.append([y,h,x,w])


            cv2.waitKey(1)


            #print(len(contours))
            #print("center", center)


    return original_image, boxes


cap=cv2.VideoCapture('/Users/itaegyeong/Desktop/2.mp4')
while True:
    _, frame=cap.read()
    mask_color(frame)

cv2.destroyAllWindows()