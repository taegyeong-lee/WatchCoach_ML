import numpy as np
import cv2

def canvas_show(our_team_point, enemy_team_point, other_point, w, h):

    img = np.zeros((w, h, 3), np.uint8)
    draw_circle(img, our_team_point, (255, 0, 0))
    draw_circle(img, enemy_team_point, (0, 0, 255))

    shrink = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow('img', shrink)
    cv2.waitKey(1)




# @brief : 테스트(시각화)용 점 찍어주는 함수
# @param : 이미지, 좌표, 색
# @return : 없음
def draw_circle(image, point, rgb):
    for i in point:
        for i2 in i:
            cv2.circle(image, (tuple)(i2), 10, rgb, -1)

