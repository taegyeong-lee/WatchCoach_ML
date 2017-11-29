import cv2
import numpy as np

video = cv2.VideoCapture('/Users/itaegyeong/Desktop/20171127_195948.mov')
mog = cv2.createBackgroundSubtractorMOG2()
termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)


while True:
    ret, frame = video.read()
    cv2.imshow('frame', frame)
    cv2.waitKey(0)
