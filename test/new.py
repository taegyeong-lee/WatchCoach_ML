import cv2

video = cv2.VideoCapture('/Users/itaegyeong/Desktop/20171127_195948.mov')

while True:
    ret, frame = video.read()

    cv2.imshow('frame',frame)
    cv2.waitKey(0)

