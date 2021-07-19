import pickle
import cv2
from cv2 import aruco
from Settings import *

CHARUCO_BOARD = aruco.CharucoBoard_create(
    CHCOL,
    CHROW,
    CHSQUARE_LEN,
    CHMARKER_LEN,
    CHARUCO_DICT)

CAM = "Camera"

corners_all = []
ids_all = []
image_size = None

cap = cv2.VideoCapture(CAM_PORT)
key = None

while key != ESC:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = aruco.detectMarkers(img, CHARUCO_DICT, parameters=CHPARAMS)
    img = aruco.drawDetectedMarkers(img, corners, ids)
    if key == SPACE and ids is not None:
        response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(corners, ids, gray, CHARUCO_BOARD)
        if response >= CHSQUARES:
            if image_size is None:
                image_size = gray.shape[::-1]
            corners_all.append(charuco_corners)
            ids_all.append(charuco_ids)
            img = aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids)
            cv2.imshow(CAM, img)
            print("Frame got successfully({})".format(len(ids_all)))
            key = cv2.waitKey(0)

    cv2.imshow(CAM, img)
    key = cv2.waitKey(TIME_DELAY)

cap.release()
cv2.destroyAllWindows()

if image_size is None:
    print('Calibration failed')
    exit()

calibration, cameraMatrix, distCoeffs, rvecs, tvecs = \
    aruco.calibrateCameraCharuco(corners_all, ids_all, CHARUCO_BOARD, image_size, None, None)

print('Camera matrix:\n', cameraMatrix)
print('Distortion coefficients:\n', distCoeffs)

with open(CALIB_FILE, 'wb') as f:
    pickle.dump((cameraMatrix, distCoeffs), f)

print('Calibration successful, saved in:', CALIB_FILE)
