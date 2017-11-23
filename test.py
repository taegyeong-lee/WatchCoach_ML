import numpy as np
import cv2

cap = cv2.VideoCapture('/Users/itaegyeong/Desktop/test.mp4')

while True:
    ret, frame = cap.read()

    if ret == True:
        width = cap.get(3)  # float
        height = cap.get(4)  # float
        frame = frame[0:100, 7:100]
        cv2.imshow('frame',frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()