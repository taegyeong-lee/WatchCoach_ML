import numpy as np
import cv2


def standard_point(point, w, h):
    weight = 407/w
    height = 720/h

    for i in point:
        for j in i:
            j[0] = int(j[0] * weight)
            j[1] = int(j[1] * height)

    return point


# 407 × 720
def canvas_show(our_team_point, enemy_team_point, other_point, w, h):

    background_img = cv2.imread('/Users/itaegyeong/Desktop/background.png') # 407x720
    cv2.imshow('back', background_img)

    if our_team_point != []:
        print(our_team_point)
        our_team_point = standard_point(our_team_point,w,h)
        print(our_team_point)
        draw_circle(background_img, our_team_point, (255, 0, 0))

    if enemy_team_point != []:
        enemy_team_point = standard_point(enemy_team_point, w, h)
        draw_circle(background_img, enemy_team_point, (0, 0, 255))

    if other_point != []:
        other_point = standard_point(other_point, w, h)
        draw_circle(background_img, other_point, (0, 255, 0))

    cv2.imshow('back',background_img)
    cv2.waitKey(1)


def draw_circle(image, point, rgb):
    for i in point:
        for j in i:
            cv2.circle(image, (tuple)(j), 7, rgb, -1)
