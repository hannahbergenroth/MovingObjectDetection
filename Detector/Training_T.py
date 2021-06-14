import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys

file_path = "../video_streams/CSMulti_AM_VIRXCam2000_T.avi"

# Load video
delay = 1
window_name = 'frame'
window_name2 = 'frame2'

kernel0 = np.ones((3, 3), np.uint8)

cap = cv2.VideoCapture(file_path)

cap.isOpened()
if not cap.isOpened():
    sys.exit()

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

fg_bg = cv2.createBackgroundSubtractorMOG2()

i = 0
while True:
    i += 1
    ret, frame = cap.read()
    if ret:
        fg_mask = fg_bg.apply(frame)



        combi = fg_mask

        # Copy the thresholded image.
        im_floodfil = combi.copy()
        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = combi.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfil, mask, (0, 0), 255)
        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfil)
        # Combine the two images to get the foreground.
        im_out2 = combi | im_floodfill_inv

        closing = cv2.morphologyEx(im_out2, cv2.MORPH_OPEN, kernel)

       # closing = cv2.dilate(closing, kernel0, iterations=1)

        print(i)

        if i == 2741:
            cv2.imwrite("data/frameT%d.jpg" % i, closing)
            print("created 21")

        _, contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            # Blob size
            if area > 150:
                x, y, w, h = cv2.boundingRect(contour)
                img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                #img2 = cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 0, 255), 1)

        cv2.imshow(window_name, frame)

        cv2.imshow('Just flood', im_out2)
        cv2.imshow('Just flood + open + close', closing)

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
cv2.destroyWindow(window_name)
