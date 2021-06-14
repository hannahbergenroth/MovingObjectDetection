import numpy as np
import cv2


def bb_intersection_over_union(box_a, box_b):
    # determine the (x, y)-coordinates of the intersection rectangle
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    # compute the area of intersection rectangle
    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = inter_area / float(box_a_area + box_b_area - inter_area)
    # return the intersection over union value
    return iou


# Color frame
frame = cv2.imread("../data/img201.jpg")
# Ground truth frame
target = cv2.imread("../ground_truth/Img0201.bmp")
# Moving object detector frame
MOD = cv2.imread("../data/frame201.jpg")
h, w, c = target.shape

# Remove shadows on GT
image = np.zeros((h, w, c), dtype="uint8")
image[np.where((target == [70, 70, 70]).all(axis=2))] = [0, 0, 0]
image[np.where((target == [255, 255, 255]).all(axis=2))] = [255, 255, 255]

gray0 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.cvtColor(MOD, cv2.COLOR_BGR2GRAY)

ret00, binary00 = cv2.threshold(gray0, 60, 255, cv2.THRESH_BINARY)
ret0, binary0 = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)

predicted = []
ground_truth = []

# Predicted bounding boxes
_, contours, _ = cv2.findContours(binary0, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    area = cv2.contourArea(contour)
    # Blob size
    if area > 100:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(MOD, (x, y), (x + w, y + h), (0, 0, 255), 1)
        predicted.append([x, y, x+w, y+h])

# Ground truth bounding boxes
_, contours, _ = cv2.findContours(binary00, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    area = cv2.contourArea(contour)
    # Blob size
    if area > 100:
        x, y, w, h = cv2.boundingRect(contour)
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        ground_truth.append([x, y, x+w, y+h])

TP = 0
FP = 0

for pred in predicted:
    if len(ground_truth) == 0:
        FP += 1
    for gt in ground_truth:
        IoU = bb_intersection_over_union(pred, gt)
        print("IoU overlap: ", "%.3f" % IoU)
        if IoU > 0.5:
            TP += 1

FP = len(predicted) - TP

FN = len(ground_truth) - TP

print("TP: ", TP, " FP: ", FP, " FN: ", FN)

cv2.imshow("Ground Truth", frame)
cv2.imshow("MOD", MOD)
cv2.waitKey()

cv2.destroyWindow("window")
