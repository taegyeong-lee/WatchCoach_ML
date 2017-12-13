import cv2
import numpy as np


def team_division(frame):
    our_team_point = []
    enemy_team_point = []
    other_point = []

    # HSV color
    boundaries = [
        (-1, [0, 140, 60], [10, 255, 255]),  # 적군 red
        (1, [80, 37, 16], [110, 189, 255]),  # 아군 blue
        (0, [50, 20, 30], [80, 255, 255])
        # (0, [20,180,0], [50,255,255])

    ]

    # hsv mask 씌우기
    opening_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for (code, lower, upper) in boundaries:
        lower = np.array(lower, dtype='uint8')
        upper = np.array(upper, dtype='uint8')

        mask = cv2.inRange(opening_hsv, lower, upper)

        _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            center_x = x + (1 / 2) * w
            center_y = y + h

            area = w * h
            if area < 350:
                continue

            # enemy team
            if code == -1:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                enemy_team_point.append([center_x, center_y])
            # our team
            elif code == 1:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                our_team_point.append([center_x, center_y])
            # other team
            elif code == 0:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                other_point.append([center_x, center_y])

    return our_team_point, enemy_team_point, other_point


def main(frame):
    our_team_point, enemy_team_point, other_point = team_division(frame)

    return frame, our_team_point, enemy_team_point, other_point