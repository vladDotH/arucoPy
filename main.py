import cv2
from ArucoFinder import ArucoFinder
import numpy as np
import pickle
from Settings import *

CAM = "Camera"
MARKERS = "Markers"

cap = cv2.VideoCapture(CAM_PORT)
finder = ArucoFinder()
key = None

with open(CALIB_FILE, 'rb') as f:
    cameraMatrix, distCoeffs = pickle.load(f)

while key != ESC:
    ret, img = cap.read()
    corners, ids = finder.find(img)
    visual = finder.visualise(img, corners, ids, cameraMatrix, distCoeffs, MARKER_LEN)
    cv2.imshow(CAM, img)
    cv2.imshow(MARKERS, visual)
    key = cv2.waitKey(TIME_DELAY)

cap.release()
cv2.destroyAllWindows()
