# Script to extract
import cv2

src1 = cv2.imread('../data/frame156.jpg')
src2 = cv2.imread('../data/img156.jpg')
video = "video_streams/CSMulti_AM_VIRXCam2000_RGB.avi"
cap = cv2.VideoCapture(video)
window_name = "window"

print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

dst = cv2.bitwise_and(src1, src2)
cv2.imshow(window_name, dst)

i = 0
while True:
    i += 1

    ret, frame = cap.read()

    if ret:

        if i == 156:

            # dst = cv2.bitwise_and(frame, src1)
            cv2.imshow(window_name, dst)
            # cv2.imwrite('data/hallo', dst)

        if cv2.waitKey(1) == ord('q'):
            break

    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

cv2.destroyWindow(window_name)
