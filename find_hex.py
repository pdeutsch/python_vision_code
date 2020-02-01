import cv2
import numpy as np
import math
import argparse

# create an array of points in the shape of a hexagon
def make_hex_shape():
    pts = []
    for ang in range(0, 355, 60):
        ang_r = math.radians(ang)
        x1 = int(100.0 * math.cos(ang_r) + 100.5)
        y1 = int(100.0 * math.sin(ang_r) + 100.5)
        pts.append([x1, y1])
    shape_np = np.array(pts, np.int32)
    shape_np = np.reshape(shape_np, (-1, 1, 2))
    return shape_np


def proc_img(shape, fname):
    print(f"---- processing file: {fname} --------")

    img = cv2.imread(fname, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (720, 480))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("orig", img)

    ret, thresh = cv2.threshold(img, 127, 255, 0)
    cv2.imshow("thresh", thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hexes = []
    for cont in contours:
        rect = cv2.boundingRect(cont)
        # only process larger areas with at least 5 points in the contour
        if len(cont) > 4 and rect[2] > 40 and rect[3] > 40:
            match = cv2.matchShapes(cont, shape, cv2.CONTOURS_MATCH_I2, 0.0)
            if match < 0.08:
                print(f"len(cont)={len(cont):3d}:  match={match:.4f}")

            if match < 0.01:
                hexes.append(cont)

    color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.drawContours(color, hexes, -1, (0,255,0), 2)
    cv2.imshow("contours", color)
    cv2.waitKey()
    cv2.destroyAllWindows()

########
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fname', help='Image filename', nargs='+')
    args = parser.parse_args()
    hex = make_hex_shape()
    for fname in args.fname:
        print(f"Processing {fname}")
        proc_img(hex, fname)
