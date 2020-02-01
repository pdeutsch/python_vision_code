import cv2
import numpy as np

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
        hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsvimg, lower_green, upper_green)

        # mask out the bottom of the image 
        mask[250:,:] = 0

        res = cv2.bitwise_and(img, img, mask=mask)
        quit_flag = show_frame(img, res, cntr)
        ret, img = cap.read()

    cap.release()

def main():
    process_video('Left_Frame.avi')
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
