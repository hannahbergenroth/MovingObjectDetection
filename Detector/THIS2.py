import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys

video = "video_streams/CSMulti_AM_VIRXCam2000_RGB.avi"
video2 = "video_streams/CSMulti_AM_VIRXCam2000_T.avi"

# Load video
delay = 1
window_name = 'frame'
window_name2 = 'frame2'

cap = cv2.VideoCapture(video)
cap2 = cv2.VideoCapture(video2)
cap.isOpened()
if not cap.isOpened():
    sys.exit()

kernel0 = np.ones((3, 3), np.uint8)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

fg_bg = cv2.createBackgroundSubtractorMOG2()
fg_bg2 = cv2.createBackgroundSubtractorMOG2()

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
        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfil, mask, (0, 0), 255)
        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfil)
        # Combine the two images to get the foreground.
        im_out2 = combi | im_floodfill_inv

        # THIS ONE UNDOO
        im_out22 = cv2.morphologyEx(im_out2, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(im_out22, cv2.MORPH_CLOSE, kernel)

        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask2 = cv2.morphologyEx(fg_mask2, cv2.MORPH_OPEN, kernel2)
        #fg_mask2 = cv2.dilate(fg_mask2, kernel = (5, 5), iterations=1)

        combined_masks = fg_mask & fg_mask2

        #dili = cv2.dilate(combined_masks, kernel=(10,10), iterations=1)

        # Copy the thresholded image.
        im_floodfill = combined_masks.copy()
        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = combined_masks.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0, 0), 255)
        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        # Combine the two images to get the foreground.
        im_out = combined_masks | im_floodfill_inv

        #dilation = cv2.dilate(closing, kernel0, iterations=1)

        _, contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            # Blob size
            if area > 150:
                x, y, w, h = cv2.boundingRect(contour)
                img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                roi = frame[y:y+h, x:x+w]
                #img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                if i == 151:
                    cv2.imwrite("../data/två.jpg", frame2)

        cv2.imshow(window_name, frame)
        cv2.imshow('Just flood + open + close', closing)
        #cv2.imshow('test', dilation)

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
cv2.destroyWindow(window_name)