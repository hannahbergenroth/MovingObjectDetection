import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys

file_path = "../video_streams/INO_MainEntrance_RGB.avi"
file_path2 = "../video_streams/INO_MainEntrance_T.avi"

# Load video
delay = 1
window_name = 'frame'
window_name2 = 'frame2'

cap = cv2.VideoCapture(file_path)
cap2 = cv2.VideoCapture(file_path2)
cap.isOpened()
if not cap.isOpened():
    sys.exit()

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

fg_bg = cv2.createBackgroundSubtractorMOG2()
fg_bg2 = cv2.createBackgroundSubtractorMOG2()

kernel_dil = np.ones((3, 3), np.uint8)

i = 0

while True:
    i += 1
    ret, frame = cap.read()
    ret2, frame2 = cap2.read()
    if ret & ret2:
        fg_mask = fg_bg.apply(frame)
        fg_mask2 = fg_bg2.apply(frame2)

        combi = fg_mask & fg_mask2

        # Copy the thresholded image.
        im_floodfil = combi.copy()
        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = combi.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        # Flood-fill from point (0, 0)
        cv2.floodFill(im_floodfil, mask, (0, 0), 255);
        # Invert flood-filled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfil)
        # Combine the two images to get the foreground.
        im_out2 = combi | im_floodfill_inv


        closing = cv2.morphologyEx(im_out2, cv2.MORPH_OPEN, kernel)


        if i == 328:
            cv2.imwrite("frame%d.jpg" % i, closing)
            print("created")

        if i == 378:
            cv2.imwrite("frame%d.jpg" % i, closing)
            print("created")

        if i == 428:
            cv2.imwrite("frame%d.jpg" % i, closing)
            print("created")

        if i == 478:
            cv2.imwrite("frame%d.jpg" % i, closing)
            print("created")

        if i == 528:
            cv2.imwrite("frame%d.jpg" % i, closing)
            print("created")

        if i == 578:
            cv2.imwrite("frame%d.jpg" % i, closing)
            print("created")

        if i == 628:
            cv2.imwrite("frame%d.jpg" % i, closing)
            print("created")

        if i == 678:
            cv2.imwrite("frame%d.jpg" % i, closing)
            print("created")

        if i == 700:
            cv2.imwrite("frame%d.jpg" % i, closing)
            print("created")

        if i == 778:
            cv2.imwrite("frame%d.jpg" % i, closing)
            print("created")

        if i == 828:
            cv2.imwrite("frame%d.jpg" % i, closing)
            print("created")

        if i == 878:
            cv2.imwrite("frame%d.jpg" % i, closing)
            print("created")

        if i == 928:
            cv2.imwrite("frame%d.jpg" % i, closing)
            print("created")

        if i == 978:
            cv2.imwrite("frame%d.jpg" % i, closing)
            print("created")

        if i == 1028:
            cv2.imwrite("frame%d.jpg" % i, closing)
            print("created")

        if i == 1078:
            cv2.imwrite("frame%d.jpg" % i, closing)
            print("created")

        if i == 1128:
            cv2.imwrite("frame%d.jpg" % i, closing)
            print("created")

        if i == 1178:
            cv2.imwrite("frame%d.jpg" % i, closing)
            print("created")

        if i == 1228:
            cv2.imwrite("frame%d.jpg" % i, closing)
            print("created")

        if i == 1278:
            cv2.imwrite("frame%d.jpg" % i, closing)
            print("created almost last")

        if i == 1328:
            cv2.imwrite("frame%d.jpg" % i, closing)
            print("created next last")

        if i == 1378:
            cv2.imwrite("frame%d.jpg" % i, closing)
            print("created last")

        _, contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            # Blob size
            if area > 250:
                x, y, w, h = cv2.boundingRect(contour)
                img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                # img2 = cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 0, 255), 1)

        cv2.imshow(window_name, frame)
        cv2.imshow('Just flood + open + close', closing)

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
cv2.destroyWindow(window_name)
