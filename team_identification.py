import numpy as np
import cv2
import time

# @brief : 팀을 구별해주는 함수 (메인)
# @param : 이미지
# @return : 1 (아군) ,-1 (적군), 0 (기타)
def team_division(image):

    img = image # or image
    # img = cv2.imread(image)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # HSV color
    boundaries = [
        ([-1], [0, 70, 50], [10, 255, 255]), # 적군 red
        ([1],[110, 50, 50],[130, 255, 255]) # 아군 blue
    ]

    list = []

    for (code, lower, upper) in boundaries:
        lower = np.array(lower, dtype='uint8')
        upper = np.array(upper, dtype='uint8')

        mask = cv2.inRange(img_hsv, lower, upper)

        count = 0
        for i in range(0, len(mask)):
            for j in mask[i]:
                if j == 255:
                    count = count + 1

        list.append([code[0], count])

        print(list)



    if list[0][1] > list[1][1] and list[0][1] > 100:
        return -1
    elif list[0][1] < list[1][1] and list[1][1] > 100:
        return 1
    else:
        return 0



