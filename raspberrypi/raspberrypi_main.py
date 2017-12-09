import cv2
import numpy as np
from raspberrypi import get_stadium_point as gsp # Get stadium's points
from raspberrypi import draw_canvas as dp # Draw camvas top view
from raspberrypi import get_object_point as gop # Get object's points
from raspberrypi import transfer_view as gtv # Transfer object's points

cap = cv2.VideoCapture('/Users/itaegyeong/Desktop/GOPR0011.mov')

if cap.isOpened():
    while True:
        ret, frame = cap.read()
        if ret is True:
            # Get stadium frame and stadium's points [ [x1,y1], [x2,y2], [x3,y3], [x4,y4] ]
            # stadium_point, stadium_frame = gsp.main(frame)
            stadium_frame = frame

            # Get trans Matrix, trans image weight and height
            #trans_matrix, trans_w, trans_h = gtv.get_trans_matrix(stadium_point[0], stadium_point[1], stadium_point[2],
            #                                                      stadium_point[3])

            # Get trans Matrix, trans image weight and height
            trans_matrix, trans_w, trans_h = gtv.get_trans_matrix([31, 160], [424, 346], [206, 38],
                                                                  [526, 147])

            # Get object's points
            find_frame, our_team_point, enemy_team_point, other_point = gop.main(stadium_frame)

            # Get trans object's points
            dst, trans_our_team_point, trans_enemy_team_point, trans_other_point = \
                gtv.trans_object_point(find_frame, our_team_point, enemy_team_point,
                                       other_point, trans_matrix, trans_w, trans_h)

            result_frame = dp.canvas_show(trans_our_team_point, trans_enemy_team_point, trans_other_point, trans_w, trans_h)

            cv2.imshow('result_frame',result_frame)
            cv2.imshow('bird eye view', dst)
            cv2.imshow('find object frame',find_frame)
            #cv2.imshow('stadium frame',stadium_frame)
            cv2.imshow('original_frame',frame)

            cv2.waitKey(1)
