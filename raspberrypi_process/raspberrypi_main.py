import cv2
import numpy as np
from raspberrypi_process import get_stadium_point as gsp # Get stadium's points
from raspberrypi_process import draw_canvas as dp # Draw camvas top view
from raspberrypi_process import get_object_point as gop # Get object's points
from raspberrypi_process import transfer_view as gtv # Transfer object's points

frame_success = False

cap = cv2.VideoCapture('/Users/itaegyeong/Desktop/GOPR0011.mov')

trans_matrix, trans_w, trans_h = None, None, None

if cap.isOpened():
    while True:
        ret, frame = cap.read()
        if ret is True:

            # Get stadium frame and stadium's points [ [x1,y1], [x2,y2], [x3,y3], [x4,y4] ]
            if frame_success is False:
                _, stadium_frame, img, stadium_point = gsp.get_stadium_line(frame)

                if stadium_frame is None:
                    continue
                else:
                    # Get trans Matrix, trans image weight and height
                    trans_matrix, trans_w, trans_h = gtv.get_trans_matrix(stadium_point[0], stadium_point[1],
                                                                          stadium_point[2],
                                                                          stadium_point[3])
                    frame_success = True


            # Get object's points
            find_frame, our_team_point, enemy_team_point, other_point = gop.main(frame)

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