import cv2
import numpy as np

video = cv2.VideoCapture('/Users/itaegyeong/Desktop/GOPR0011.MP4')


def callback(x):
    pass

cv2.namedWindow('image')

ilowH = 0
ihighH = 179
ilowS = 0
ihighS = 255
ilowV = 0
ihighV = 255

# create trackbars for color change
cv2.createTrackbar('lowH','image',ilowH,179,callback)
cv2.createTrackbar('highH','image',ihighH,179,callback)
cv2.createTrackbar('lowS','image',ilowS,255,callback)
cv2.createTrackbar('highS','image',ihighS,255,callback)
cv2.createTrackbar('lowV','image',ilowV,255,callback)
cv2.createTrackbar('highV','image',ihighV,255,callback)


while(True):
    # grab the frame
    ret, frame = video.read()
    original = frame.copy()

    # get trackbar positions
    ilowH = cv2.getTrackbarPos('lowH', 'image')
    ihighH = cv2.getTrackbarPos('highH', 'image')
    ilowS = cv2.getTrackbarPos('lowS', 'image')
    ihighS = cv2.getTrackbarPos('highS', 'image')
    ilowV = cv2.getTrackbarPos('lowV', 'image')
    ihighV = cv2.getTrackbarPos('highV', 'image')

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([ilowH, ilowS, ilowV])
    higher_hsv = np.array([ihighH, ihighS, ihighV])
    mask = cv2.inRange(hsv, lower_hsv, higher_hsv)
    frame = cv2.bitwise_and(frame, frame, mask=mask)

    # show thresholded image
    # 이미지 축소
    shrink = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    shrink2 = cv2.resize(mask, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    cv2.imshow('image', shrink)
    cv2.imshow('mask',shrink2)

    k = cv2.waitKey(0) & 0xFF # large wait time to remove freezing
    if k == 113 or k == 27:
        break
