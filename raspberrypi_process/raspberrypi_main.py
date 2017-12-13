import cv2
import numpy as np
from raspberrypi_process import get_stadium_point as gsp # Get stadium's points
from raspberrypi_process import draw_canvas as dp # Draw camvas top view
from raspberrypi_process import get_object_point as gop # Get object's points
from raspberrypi_process import transfer_view as gtv # Transfer object's points

frame_success = False
background_img = cv2.imread('./background.png')

cap = cv2.VideoCapture('/dev/video0')

trans_matrix, trans_w, trans_h = None, None, None

our_defence_flag = False
enemy_defence_flag = False

if cap.isOpened():
    while True:
        ret, frame = cap.read()
        if ret is True:

            # Get stadium frame and stadium's points [ [x1,y1], [x2,y2], [x3,y3], [x4,y4] ]
            if frame_success is False:
                background, stadium_frame, img, stadium_point = gsp.get_stadium_line(frame)

                if stadium_frame is None:
                    continue
                else:
                    # Get trans Matrix, trans image weight and height
                    stadium_point.sort(key=lambda tup: tup[1])
                    print(stadium_point)
                    trans_matrix, trans_w, trans_h = gtv.get_trans_matrix(stadium_point[0], stadium_point[2],
                                                                          stadium_point[1],
                                                                          stadium_point[3])
                    frame_success = True
                    bakground = cv2.bitwise_and(background, background, mask=stadium_frame)
                    background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

            # Get object's points
            frame_and = cv2.bitwise_and(frame, frame, mask=stadium_frame)
            gray = cv2.cvtColor(frame_and, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(background, gray)
            ret, th1 = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
            kernel = np.ones((3, 3), np.uint8)

            th1 = cv2.erode(th1, kernel, iterations=1)
            th1 = cv2.dilate(th1, kernel, iterations=4)
            th1 = cv2.erode(th1, kernel, iterations=2)
            cv2.imshow('diff', th1)

            t1 = time.time()

            find_frame, our_team_point, enemy_team_point, other_point = gop.main(frame)

            t2 = time.time()
            print(t2 - t1)
            # Get trans object's points
            dst, trans_our_team_point, trans_enemy_team_point, trans_other_point = \
                gtv.trans_object_point(find_frame, our_team_point, enemy_team_point,
                                       other_point, trans_matrix, trans_w, trans_h)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('d'):
                if our_defence_flag is False:
                    our_defence_flag = True
                else:
                    our_defence_flag = False

            if key == ord('f'):
                if enemy_defence_flag is False:
                    enemy_defence_flag = True
                else:
                    enemy_defence_flag = False

            if our_defence_flag is True and enemy_defence_flag is False:
                result_frame = dp.canvas_show(trans_our_team_point, trans_enemy_team_point, trans_other_point,
                                              trans_w, trans_h, 1)
            if enemy_defence_flag is True and our_defence_flag is False:
                result_frame = dp.canvas_show(trans_our_team_point, trans_enemy_team_point, trans_other_point,
                                              trans_w, trans_h, -1)
            if enemy_defence_flag is True and our_defence_flag is True:
                result_frame = dp.canvas_show(trans_our_team_point, trans_enemy_team_point, trans_other_point,
                                              trans_w, trans_h, 2)

            if our_defence_flag is False and enemy_defence_flag is False:
                result_frame = dp.canvas_show(trans_our_team_point, trans_enemy_team_point, trans_other_point,
                                              trans_w, trans_h, 0)

            # result_frame = dp.canvas_show(background_img, trans_our_team_point, trans_enemy_team_point, trans_other_point, trans_w, trans_h)

            # cv2.imshow('backgorund',background_img)
            cv2.imshow('img', img)
            cv2.imshow('result_frame', result_frame)
            # cv2.imshow('bird eye view', dst)
            find_frame = cv2.bitwise_and(find_frame, find_frame, mask=stadium_frame)
            cv2.imshow('find object frame', find_frame)
            # cv2.imshow('stadium frame',stadium_frame)
            # cv2.imshow('original_frame',frame)


def test():

    our_defence_flag = False
    enemy_defence_flag = False

    cap = cv2.VideoCapture('/Users/itaegyeong/Desktop/GOPR0011.mov')

    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if ret is True:


                # Get trans Matrix, trans image weight and height
                trans_matrix, trans_w, trans_h = gtv.get_trans_matrix([23,164],[432,358], [205,34]
                                                                      ,[532,142])

                # Get object's points
                find_frame, our_team_point, enemy_team_point, other_point = gop.main(frame)

                # Get trans object's points
                dst, trans_our_team_point, trans_enemy_team_point, trans_other_point = \
                    gtv.trans_object_point(find_frame, our_team_point, enemy_team_point,
                                           other_point, trans_matrix, trans_w, trans_h)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('d'):
                    if our_defence_flag is False:
                        our_defence_flag = True
                    else:
                        our_defence_flag = False

                if key == ord('f'):
                    if enemy_defence_flag is False:
                        enemy_defence_flag = True
                    else:
                        enemy_defence_flag = False

                if our_defence_flag is True and enemy_defence_flag is False:
                    result_frame = dp.canvas_show(trans_our_team_point, trans_enemy_team_point, trans_other_point,
                                                  trans_w,trans_h, 1)
                if enemy_defence_flag is True and our_defence_flag is False:
                    result_frame = dp.canvas_show(trans_our_team_point, trans_enemy_team_point, trans_other_point,
                                                  trans_w,trans_h, -1)
                if enemy_defence_flag is True and our_defence_flag is True:
                    result_frame = dp.canvas_show(trans_our_team_point, trans_enemy_team_point, trans_other_point,
                                                  trans_w,trans_h, 2)

                if our_defence_flag is False and enemy_defence_flag is False:
                    result_frame = dp.canvas_show(trans_our_team_point, trans_enemy_team_point, trans_other_point,
                                                  trans_w, trans_h, 0)

                cv2.imshow('result_frame', result_frame)
                cv2.imshow('bird eye view', dst)
                cv2.imshow('find object frame', find_frame)
                # cv2.imshow('stadium frame',stadium_frame)
                cv2.imshow('original_frame', frame)


test()
