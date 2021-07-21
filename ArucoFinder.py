import cv2
from cv2 import aruco
import numpy as np
from Settings import *


class ArucoFinder:
    def __init__(self,
                 dict=aruco.Dictionary_get(aruco.DICT_6X6_1000),
                 params=aruco.DetectorParameters_create()):
        self.dict = dict
        self.params = params

    def find(self, img):
        corners, ids, rejected = aruco.detectMarkers(img, self.dict, parameters=self.params)
        return corners, ids

    @staticmethod
    def center(marker):
        p1, p2 = marker[0][0], marker[0][2]
        return int(p1[0] + p2[0]) // 2, int(p1[1] + p2[1]) // 2

    @staticmethod
    def dist(m1, m2):
        return np.linalg.norm(m1 - m2)

    def visualise(self, img, corners, ids, camMat, distCoeffs, markerLen):
        markers = len(corners)
        if markers > 0:
            y = img.shape[0] + SPAN
            img = cv2.copyMakeBorder(img, 0, (markers + 1 + 1 * (markers == 2)) * SPAN, 0, 0, cv2.BORDER_CONSTANT, value=WHITE)
            aruco.drawDetectedMarkers(img, corners, ids)

            ids = np.array(ids).flatten()
            markersDict = dict(zip(ids, corners))
            coords = []
            ids.sort()
            for id in ids:
                rvec, tvec, objPoints = aruco.estimatePoseSingleMarkers(markersDict[id], markerLen, camMat, distCoeffs)
                tvec = np.array(tvec).flatten()
                coords.append(tvec)
                cv2.putText(img, "Id: {}; Pos: x:{} ; y:{} ; z:{}".format(id, *np.round(tvec, PRECISION)),
                            (0, y), FONT, TEXT_SCALE, BLACK)
                y += SPAN

            if markers == 2:
                cv2.line(img, *list(map(ArucoFinder.center, corners)), RED, 2)
                cv2.putText(img,
                            "Distance between {} and {} : {} meters"
                            .format(ids[0], ids[1], np.round(ArucoFinder.dist(*coords), PRECISION)),
                            (0, y), FONT, TEXT_SCALE, BLACK)

        return img
