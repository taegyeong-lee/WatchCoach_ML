
import numpy as np
import cv2


# 65 30 40
green = np.uint8([[[25,28,90]]])
hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)

print(hsv_green)


# [[[178 155  94]]]
#[[[176 166 114]]]
#
# 160, 150, 80
#  [[[150  92  50]]]
# [[[142 186  37]]]
#
#
# [[[  0 166 114]]]