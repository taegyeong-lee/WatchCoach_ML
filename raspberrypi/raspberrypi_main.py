import cv2
import numpy as np
from . import get_stadium_point as gsp # Get stadium's points
from . import draw_canvas as dp # Draw camvas top view
from . import get_object_point as gop # Get object's points
from . import transfer_view as gtv # Transfer object's points


cap = cv2.VideoCapture('/dev/video0')

if cap.isOpened():
    while True:
        ret, frame = cap.read()

        # Get stadium frame and stadium's points [ [x1,y1], [x2,y2], [x3,y3], [x4,y4] ]
        stadium_point, stadium_frame = gsp.main(frame)

        # Get trans Matrix, trans image weight and height
        trans_matrix, trans_w, trans_h = gtv.get_trans_matrix(stadium_point[0], stadium_point[1], stadium_point[2],
                                                              stadium_point[3])

        # Get object's points
        our_team_point, enemy_team_point, other_point = gop.main(stadium_frame)

        # Get trans object's points
        dst, trans_our_team_point, trans_enemy_team_point, trans_other_point = \
            gtv.trans_object_point(stadium_frame, our_team_point, enemy_team_point,
                                   other_point, trans_matrix, trans_w, trans_h)


        result_frame = dp.canvas_show(our_team_point, enemy_team_point, our_team_point)


        cv2.imshow('result_frame',result_frame)
        cv2.imshow('stadium frame',stadium_frame)
        cv2.imshow('original_frame',frame)

        cv2.waitKey(1)








        

