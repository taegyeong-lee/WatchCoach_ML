import cv2
import numpy as np

import team_identification as ti
from trash import opencv_object_detection as oc

cap = cv2.VideoCapture('/Users/itaegyeong/Desktop/test.mp4')


# take first frame of the video
ret,frame = cap.read()


# setup initial location of window


trans_frame, boxes = oc.mask_color(frame)


r,h,c,w = boxes[8][0],boxes[8][1],boxes[8][2],boxes[8][3]
print(r,h,c,w)


# simply hardcoded the values
track_window = (c,r,w,h)
# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]


hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))

cv2.imshow('m',mask)
cv2.waitKey(1)


roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])


cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )


while(1):
    ret ,frame = cap.read()
    if ret == True:
        copy = frame.copy()


        trans_frame, boxes = oc.mask_color(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        cv2.imshow('ㅇ',dst)


        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        # Draw it on image
        x,y,w,h = track_window

        result = ti.team_division(frame[y:y+h, x:x+w])


        img2 = cv2.rectangle(copy, (x, y), (x + w, y + h), 255, 2)

        for i in boxes:
            img2 = cv2.rectangle(copy, (i[2], i[0]), (i[2]+i[3], i[0]+i[1]), 255, 2)

        cv2.imshow('img2',img2)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k)+".jpg",img2)
    else:
        break
cv2.destroyAllWindows()
cap.release()