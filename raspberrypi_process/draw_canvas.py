import numpy as np
import cv2


def standard_point(point, w, h):
    weight = 389/w
    height = 720/h

    for i in point:
        for j in i:
            j[0] = int(j[0] * weight)
            j[1] = int(j[1] * height)

    return point


# 1 = our team, -1 = enemy team
def canvas_show(our_team_point, enemy_team_point, other_point, w, h, defecne_flag):

    background_img = cv2.imread('/Users/itaegyeong/Desktop/background.png') # 407x720

    if our_team_point != []:
        our_team_point = standard_point(our_team_point,w,h)
        draw_circle(background_img, our_team_point, (255, 0, 0))
        if defecne_flag == 1:
            draw_defence_range(background_img,our_team_point,(255, 255, 47))

    if enemy_team_point != []:
        enemy_team_point = standard_point(enemy_team_point, w, h)
        draw_circle(background_img, enemy_team_point, (0, 0, 255))
        if defecne_flag == -1:
            draw_defence_range(background_img, enemy_team_point, (0, 0, 255))

    if enemy_team_point != [] and our_team_point != [] and defecne_flag == 2:
        draw_defence_range(background_img, our_team_point, (255, 255, 47))
        draw_defence_range(background_img, enemy_team_point, (0, 0, 255))

    if other_point != []:
        other_point = standard_point(other_point, w, h)
        draw_circle(background_img, other_point, (0, 255, 0))

    return background_img


def draw_circle(image, point, rgb):
    for i in point:
        for j in i:
            cv2.circle(image, (tuple)(j), 12, rgb, -1)


def draw_defence_range(image, point, rgb):
    for i in point:
        for j in i:
            overlay = image.copy()
            cv2.circle(overlay, (tuple)(j), 60, rgb, -1)
            opacity = 0.2
            cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0, image)