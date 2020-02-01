import cv2
import numpy as np
import math

cntr = 0

def make_hex_shape():
    pts = []
    for ang in [0, 60, 120, 180]:
        ang_r = math.radians(ang)
        x1 = int(100.0 * math.cos(ang_r) + 100.5)
        y1 = int(100.0 * math.sin(ang_r) + 100.5)
        pts.append([x1, y1])

    for ang in [180, 120, 60, 0]:
        ang_r = math.radians(ang)
        x1 = int(92.0 * math.cos(ang_r) + 100.5)
        y1 = int(92.0 * math.sin(ang_r) + 100.5)
        pts.append([x1, y1])
    shape_np = np.array(pts, np.int32)
    shape_np = np.reshape(shape_np, (-1, 1, 2))
    return shape_np


def proc_img(shape, mask, img):
    global cntr
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hexes = []

    cntr += 1
    if cntr == 100:
        print(f"Got contours, count: {len(contours)}")
        print(contours)
    #img = cv2.drawContours(img, contours, -1, (0,255,255), 2)

    for cont in contours:
        rect = cv2.boundingRect(cont)
        # only process larger areas with at least 5 points in the contour
        if True or (len(cont) > 4 and rect[2] > 40 and rect[3] > 40):
            match = cv2.matchShapes(cont, shape, cv2.CONTOURS_MATCH_I2, 0.0)
            if cntr == 100:
                print(f"Match: {match}")

            if match < 0.08:
                print(f"len(cont)={len(cont):3d}:  match={match:.4f}")

            if match < 10:
                hexes.append(cont)

    img = cv2.drawContours(img, hexes, -1, (0,0,255), 2)

    img = cv2.drawContours(img, [shape], -1, (0,255,0), 2)
    return img

def show_frame(img, mask, cntr):
    cv2.imshow('orig', img)
    cv2.imshow('mask', mask)
    ch = cv2.waitKey(5) & 0xff
    if ch == ord('s'):
        cv2.imwrite(f"frame_{cntr:03d}.png", img)
    if ch == 27:
        return True
    return False

def process_video(fname):
    shape = make_hex_shape()
    cap = cv2.VideoCapture(fname)
    #hsv max: 89, 255, 255
    #hsv min: 78, 231, 240
    upper_green = np.array([89,255,255])
    lower_green = np.array([78,150,150])
    cv2.namedWindow('orig')
    cv2.namedWindow('mask')
    cv2.moveWindow('orig', 0, 0)
    cv2.moveWindow('mask', 600, 0)
    quit_flag = False
    cntr = 0

    ret, img = cap.read()
    while ret and not quit_flag:
        cntr += 1
        img = cv2.resize(img, None, fx=0.5, fy=0.5)
        hsvimg = cv2.GaussianBlur(img, (5,5), 0)
        hsvimg = cv2.cvtColor(hsvimg, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsvimg, lower_green, upper_green)

        # mask out the bottom of the image 
        mask[250:,:] = 0

        img = proc_img(shape, mask, img)

        res = cv2.bitwise_and(img, img, mask=mask)
        quit_flag = show_frame(img, res, cntr)
        ret, img = cap.read()

    cap.release()

def main():
    process_video('Left_Frame.avi')
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
