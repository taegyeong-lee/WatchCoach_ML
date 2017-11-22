import cv2
import numpy as np

image_path = "/Users/itaegyeong/Desktop/testblue.png"
img = cv2.imread(image_path)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_blue = np.array([110, 51, 51], dtype='uint8')
upper_blue = np.array([130, 255, 255], dtype='uint8')

mask = cv2.inRange(img_hsv, lower_blue, upper_blue)


count = 0
for i in range(0, len(mask)):
    for j in mask[i]:
        if j == 255:
            count = count + 1

print(count)

cv2.imshow('frame',mask)


key = cv2.waitKey(0)

if key == ord('s'):
    cv2.destroyAllWindows()
