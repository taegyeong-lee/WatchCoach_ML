import numpy as np
import cv2

def canvas_show(our_team_point, enemy_team_point, other_point, w, h):

    img = np.zeros((h, w, 3), np.uint8)
    draw_circle(img, our_team_point, (255, 0, 0))
    draw_circle(img, enemy_team_point, (0, 0, 255))


    cv2.imshow('img', img)
    cv2.waitKey(1)




# @brief : 테스트(시각화)용 점 찍어주는 함수
# @param : 이미지, 좌표, 색
# @return : 없음
def draw_circle(image, point, rgb):
    for i in point:
        for i2 in i:
            cv2.circle(image, (tuple)(i2), 10, rgb, -1)

def draw_circle2(image, point, rgb):

    point2 = []
    for i in point:
        cv2.circle(image, (int(i[0]),int(i[1])), 3, rgb, -1)


def draw_circle3(point, rgb):
    img = np.zeros((360, 640, 3), np.uint8)

    point2 = []
    for i in point:
        cv2.circle(img, (int(i[0]), int(i[1])), 3, rgb, -1)

    cv2.imshow('img', img)
    cv2.waitKey(1)