
import numpy as np
import cv2


# 65 30 40
green = np.uint8([[[0,0,255]]])
hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)

print(hsv_green)

