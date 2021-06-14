import numpy as np
import cv2

video = "video_streams/INO_MainEntrance_RGB.avi"
video2 = "video_streams/INO_MainEntrance_T.avi"

cap = cv2.VideoCapture(video)
cap2 = cv2.VideoCapture(video2)

window_name = 'frame'
window_name2 = 'frame2'

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

kernel0 = np.ones((3, 3), np.uint8)

# Background subtraction function CV
fg_bg = cv2.createBackgroundSubtractorMOG2()
fg_bg2 = cv2.createBackgroundSubtractorMOG2()

# 1. Save the first image as reference
amount_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
# cap.set(cv2.CAP_PROP_POS_FRAMES, 1)

i = 0
while True:
    i += 1
    # Capture frame-by-frame
    ret, frame = cap.read()
    ret2, frame2 = cap2.read()

    if ret & ret2:
        fg_mask = fg_bg.apply(frame)
        fg_mask2 = fg_bg2.apply(frame2)

        fg_mask2 = cv2.dilate(fg_mask2, kernel0, iterations=1)
        combi = fg_mask & fg_mask2

        # Copy the thresholded image.
        im_floodfil = combi.copy()
        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = combi.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        # Flood-fill from point (0, 0)
        cv2.floodFill(im_floodfil, mask, (0, 0), 255)
        # Invert flood-filled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfil)
        # Combine the two images to get the foreground.
        im_out2 = combi | im_floodfill_inv

        closing = cv2.morphologyEx(im_out2, cv2.MORPH_OPEN, kernel)
        #closing = cv2.dilate(closing, kernel0, iterations=1)

        _, contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            # Blob size
            if area > 100:
                x, y, w, h = cv2.boundingRect(contour)
                img = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
                # roi = frame[y:y-10+h+5,x:x-8+w+10]

        # Display the resulting frame
        cv2.imshow(window_name, frame)
        cv2.imshow('Background Subtraction', closing)

        if cv2.waitKey(1) == ord('q'):
            break

    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)

cv2.destroyWindow(window_name)
