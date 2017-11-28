import numpy as np
import cv2

cap = cv2.VideoCapture('/Users/itaegyeong/Desktop/camshifttest.mov')

ret, frame = cap.read()
r, h, c, w = 80, 5, 159, 5   # simply hardcoded the values

r2, h2, c2, w2 = 300, 5, 417, 5   # simply hardcoded the values

track_window = (c, r, w, h)
roi = frame[r:r + h, c:c + w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

track_window2 = (c2, r2, w2, h2)
roi2 = frame[r2:r2 + h2, c2:c2 + w2]
hsv_roi2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV)


mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)


mask2 = cv2.inRange(hsv_roi2, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist2 = cv2.calcHist([hsv_roi2], [0], mask2, [180], [0, 180])
cv2.normalize(roi_hist2, roi_hist2, 0, 255, cv2.NORM_MINMAX)


term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while (1):
    ret, frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        hsv2 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst2 = cv2.calcBackProject([hsv2], [0], roi_hist2, [0, 180], 1)

        # apply meanshift to get the new location

        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        ret2, track_window2 = cv2.CamShift(dst2, track_window2, term_crit)
        print(ret,track_window)
        print("2",ret2, track_window2)

        # Draw it on image
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame, [pts], True, 255, 2)

        pts2 = cv2.boxPoints(ret2)
        pts2 = np.int0(pts2)
        img3 = cv2.polylines(frame, [pts2], True, 255, 2)

        cv2.imshow('img2', img2)

        k = cv2.waitKey(1) & 0xff

        if k == 27:
            break
        else:
            cv2.imwrite(chr(k) + ".jpg", img2)
    else:
        break

cv2.destroyAllWindows()
cap.release()