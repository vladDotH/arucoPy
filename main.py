import cv2
from ArucoFinder import ArucoFinder

CAM_PORT = 1
ESC = 27
TIME_DELAY = 10
CAM = "Camera"
MARKERS = "Markers"

cap = cv2.VideoCapture(CAM_PORT)
finder = ArucoFinder()
key = None

while key != ESC:
    ret, img = cap.read()
    corners, ids = finder.find(img)
    visual = finder.visualise(img, corners, ids)
    cv2.imshow(CAM, img)
    cv2.imshow(MARKERS, visual)
    key = cv2.waitKey(TIME_DELAY)

cap.release()
cv2.destroyAllWindows()
