import cv2
import numpy as np

from raspberrypi import get_stadium_point as gsp # Get stadium's points
from raspberrypi import draw_canvas as dp # Draw camvas top view
from raspberrypi import get_object_point as gop # Get object's points
from raspberrypi import transfer_view as gtv # Transfer object's points


def team_division(frame):
    our_team_point = []
    enemy_team_point = []
    other_point = []

    # HSV color
    boundaries = [
        (-1, [0, 140, 70], [10, 255, 255]),  # 적군 red
        (1, [90, 60, 130], [105, 255, 255])  # 아군 blue
    ]

    # hsv mask 씌우기
    opening_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for (code, lower, upper) in boundaries:
        lower = np.array(lower, dtype='uint8')
        upper = np.array(upper, dtype='uint8')

        mask = cv2.inRange(opening_hsv, lower, upper)
        cv2.imshow('masks', mask)

        _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)

            area = w * h
            if area < 500:
                continue

            # enemy team
            if code == -1:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                enemy_team_point.append([x,y])
            # our team
            elif code == 1:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                our_team_point.append([x, y])
            # other team
            elif code == 0:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                other_point.append([x,y])

    cv2.imshow('frame',frame)
    cv2.waitKey(0)
    return our_team_point,enemy_team_point,other_point


def main(frame):
    # Morphology의 opening, closing을 통해서 노이즈제거
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel, iterations=1)
    our_team_point, enemy_team_point, other_point = team_division(opening)
    return our_team_point, enemy_team_point, other_point


def test():
    cap = cv2.VideoCapture('/Users/itaegyeong/Desktop/GOPR0011.mov')

    # Get trans Matrix, trans image weight and height
    trans_matrix, trans_w, trans_h = gtv.get_trans_matrix([206,47], [512,149], [65, 159],
                                                          [421,332])

    if cap.isOpened():
        while True:
            ret, frame = cap.read()

            # Morphology의 opening, closing을 통해서 노이즈제거
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel, iterations=1)
            cv2.imshow('opening', opening)

            our_team_point, enemy_team_point, other_point = team_division(opening)

            # Get trans object's points
            dst, trans_our_team_point, trans_enemy_team_point, trans_other_point = \
                gtv.trans_object_point(frame, our_team_point, enemy_team_point,other_point, trans_matrix, trans_w, trans_h)
            gtv.trans_object_point(frame, our_team_point, enemy_team_point, other_point, trans_matrix, trans_w, trans_h)

            cv2.imshow('dst',dst)


            result_frame = dp.canvas_show(our_team_point, enemy_team_point, our_team_point, trans_w, trans_h)

test()
