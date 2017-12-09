import cv2
import numpy as np

our_point = []
enemy_point = []
other_point = []


frame_point = []


def draw_circle(event,x,y,flags,param):

    if event == cv2.EVENT_LBUTTONDBLCLK:

        global mouseX, mouseY
        mouseX,mouseY = x,y
        #print(mouseX,mouseY)

        cv2.circle(cams2,(x,y),7,(0,0,255),-1)
        our_point.append([mouseX, mouseY])

    elif event == cv2.EVENT_RBUTTONDBLCLK:

        mouseX,mouseY = x,y
        #print(mouseX,mouseY)

        cv2.circle(cams2, (x, y), 7, (0, 255, 0), -1)
        enemy_point.append([mouseX,mouseY])


    cv2.imshow('result',cams2)
    cv2.imwrite(str(frame_count) + '.jpg', cams2)


cam1 = cv2.VideoCapture('/Users/itaegyeong/Desktop/video/cam1_left.mov')
cam2 = cv2.VideoCapture('/Users/itaegyeong/Desktop/video/cam2_left.mov')
cam3 = cv2.VideoCapture('/Users/itaegyeong/Desktop/video/cam3_right.mov')
cam4 = cv2.VideoCapture('/Users/itaegyeong/Desktop/video/cam4_right.mov')

background = cv2.imread('/Users/itaegyeong/Desktop/background.png')


cv2.namedWindow('image')

cv2.setMouseCallback('image', draw_circle)

# 640 x 360
frame_count = 0

while True:

    _, cam1_frame = cam1.read()
    _, cam2_frame = cam2.read()
    _, cam3_frame = cam3.read()
    _, cam4_frame = cam4.read()

    left_cam = np.concatenate((cam1_frame, cam2_frame), axis=0)
    right_cam = np.concatenate((cam3_frame, cam4_frame), axis=0)

    cams = np.concatenate((left_cam,background),axis=1)
    cams2 = np.concatenate((cams, right_cam), axis=1)

    our_point = []
    enemy_point = []


    cv2.imshow('image',cams2)

    frame_point.append([our_point,enemy_point])
    print(frame_point)

    frame_count = frame_count + 1
    cv2.waitKey(0)


    print(frame_count)
