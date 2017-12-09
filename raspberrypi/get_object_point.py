import cv2
import numpy as np

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
            center_x = x+(1/2)*w
            center_y = y + h


            area = w * h
            if area < 500:
                continue

            # enemy team
            if code == -1:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                enemy_team_point.append([center_x,center_y])
            # our team
            elif code == 1:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                our_team_point.append([center_x, center_y])
            # other team
            elif code == 0:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                other_point.append([center_x,center_y])

    cv2.imshow('frame',frame)
    cv2.waitKey(1)
    return our_team_point,enemy_team_point,other_point


def main(frame):
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel, iterations=1)
    our_team_point, enemy_team_point, other_point = team_division(opening)

    return our_team_point, enemy_team_point, other_point
