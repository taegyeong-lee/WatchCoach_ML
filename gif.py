import cv2

cap = cv2.VideoCapture('/Users/itaegyeong/Desktop/multi object detection soccer.mp4')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('soccer_output.mp4',fourcc, 60.0, (640,360))

if cap.isOpened():
    while True:
        ret, frame = cap.read()
        if ret is True:
            cv2.imshow('frame',frame)
            cv2.waitKey(1)
            out.write(frame)


