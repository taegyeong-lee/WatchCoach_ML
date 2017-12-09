import cv2
import numpy as np
import transfer_view as tv

our_point = []
enemy_point = []
other_point = []


frame_point = []


def draw_circle(event,x,y,flags,param):

    if event == cv2.EVENT_LBUTTONDBLCLK:

        global mouseX, mouseY
        mouseX,mouseY = x,y
        #print(mouseX,mouseY)

        cv2.circle(cam1_frame,(x,y),7,(0,0,255),-1)
        our_point.append([mouseX, mouseY])

    elif event == cv2.EVENT_RBUTTONDBLCLK:

        mouseX,mouseY = x,y
        #print(mouseX,mouseY)

        cv2.circle(cam1_frame, (x, y), 7, (0, 255, 0), -1)
        enemy_point.append([mouseX,mouseY])


    cv2.imshow('result',cam1_frame)
    cv2.imwrite(str(frame_count) + '.jpg', cam1_frame)


cam1 = cv2.VideoCapture('/Users/itaegyeong/Desktop/video/cam1_left.mov')

cv2.namedWindow('image')

cv2.setMouseCallback('image', draw_circle)

# 640 x 360
frame_count = 0

while True:

    _, cam1_frame = cam1.read()
    frame_count = frame_count + 1

    if frame_count % 5 != 0:
        continue

    our_point = []
    enemy_point = []


    cv2.imshow('image',cam1_frame)

    frame_point.append([our_point,enemy_point])
    print(frame_point)


    cv2.waitKey(0)

    print(frame_count)
