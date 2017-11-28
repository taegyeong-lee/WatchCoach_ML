import cv2, sys
import numpy as np

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture()
cap.open("rtmp://172.16.101.160:1935/live/stream")

if not cap.open:
    print("a")

while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    cv2.waitKey(1)


