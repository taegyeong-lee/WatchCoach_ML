import numpy as np
import cv2
import random

# 407 × 720

def canvas_show(our_team_point, enemy_team_point, other_point, w, h):

    weight = 407/w
    height = 720/h

    for i in our_team_point:
        print("i",i)


    background_img = cv2.imread('/Users/itaegyeong/Desktop/background.png') # 407x720
    cv2.imshow('back', background_img)

    draw_circle(background_img, our_team_point, (255, 0, 0))
    draw_circle(background_img, enemy_team_point, (0, 0, 255))
    draw_circle(background_img, other_point, (0, 255, 0))


    cv2.waitKey(1)


def draw_circle(image, point, rgb):
    for i in point:
            cv2.circle(image, (tuple)(i), 3, rgb, -1)


