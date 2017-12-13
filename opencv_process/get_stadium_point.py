import numpy as np
import cv2


def get_stadium_points(lines, h, w):
    pt = []
    l = len(lines)
    for i in range(0, l):
        for j in range(i + 1, l):

            r1 = lines[i][0][0]
            r2 = lines[j][0][0]

            theta1 = lines[i][0][1]
            theta2 = lines[j][0][1]

            ct1 = np.cos(theta1)
            st1 = np.sin(theta1)
            ct2 = np.cos(theta2)
            st2 = np.sin(theta2)

            d = ct1 * st2 - st1 * ct2
            if d != 0 and abs(theta1 - theta2) > 0.7:
                x = int((st2 * r1 - st1 * r2) / d)
                y = int((-ct2 * r1 + ct1 * r2) / d)

                if y > 0 and x > 0:
                    if y < h + 1 and x < w + 1:
                        pt.append((x, y))
    return pt


def get_stadium_line(frame):
    kernel = np.ones((3, 3), np.uint8)

    points = []

    background = frame
    # cv2.imshow('back', background)
    img = background.copy()
    background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    h, w = img.shape[:2]
    mk = np.zeros((h + 2, w + 2), np.uint8)

    mask = cv2.inRange(img, np.array([140, 140, 140]), np.array([220, 220, 220]))
    # cv2.imshow('mask', mask)
    mop = cv2.dilate(mask, kernel, iterations=3)
    mop = cv2.erode(mop, kernel, iterations=1)
    cv2.imshow('mop', mop)
    rev = cv2.bitwise_not(mop)
    cv2.floodFill(rev, mk, (0, 0), (0, 0, 0), 30, 30, 4)
    cv2.floodFill(rev, mk, (0, h-1), (0, 0, 0), 30, 30, 4)
    cv2.floodFill(rev, mk, (w-1, 0), (0, 0, 0), 30, 30, 4)
    cv2.floodFill(rev, mk, (w-1, h-1), (0, 0, 0), 30, 30, 4)
    #cv2.imshow('rev',rev)
    cv2.waitKey(1)
    edges = cv2.Canny(rev, 50, 150, apertureSize=3)
    # cv2.imshow("edges", edges)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 65)
    if lines is not None:
        if len(lines) > 3 and len(lines) < 20:
            points = get_stadium_points(lines, h, w)
            print(points)
            for l in lines:
                for rho, theta in l:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))

                    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    l_pts = len(points)

    dup = []

    for i in range(0, l_pts):
        for j in range(i + 1, l_pts):
            d = np.sqrt((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2)
            if (d < 100):
                dup.append(j)
    dup = list(set(dup))

    for k in dup:
        points[k] = (0, 0)
    points = list(set(points))

    if (0, 0) in points:
        points.remove((0, 0))

    for x, y in points:
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)



    if (len(points) == 4):

        return frame, rev, img, points

    return None, None, None, None