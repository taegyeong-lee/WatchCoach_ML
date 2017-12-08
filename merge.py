import cv2
import numpy as np

cam1 = cv2.VideoCapture('/Users/itaegyeong/Desktop/video/cam1_left.mov')
cam2 = cv2.VideoCapture('/Users/itaegyeong/Desktop/video/cam2_left.mov')
cam3 = cv2.VideoCapture('/Users/itaegyeong/Desktop/video/cam3_right.mov')
cam4 = cv2.VideoCapture('/Users/itaegyeong/Desktop/video/cam4_right.mov')

background = cv2.imread('/Users/itaegyeong/Desktop/background.png')


# 640 x 360

while True:

    ret, cam1_frame = cam1.read()
    ret, cam2_frame = cam2.read()
    ret, cam3_frame = cam3.read()
    ret, cam4_frame = cam4.read()

    left_cam = np.concatenate((cam1_frame, cam2_frame), axis=0)
    right_cam = np.concatenate((cam3_frame, cam4_frame), axis=0)

    cams = np.concatenate((left_cam,background),axis=1)
    cams2 = np.concatenate((cams, right_cam), axis=1)
    cv2.imshow('vis',cams2)
    cv2.waitKey(0)

