import cv2
import math
import numpy as np


class ArucoFinder:
    SPAN = 30
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    BLACK = (0, 0, 0)
    WHITE = (0xff, 0xff, 0xff)
    RED = (0, 0, 0xff)
    TEXT_SCALE = 0.5

    def __init__(self,
                 dict=cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_1000),
                 params=cv2.aruco.DetectorParameters_create()):
        self.dict = dict
        self.params = params

    def find(self, img):
        corners, ids, rejected = cv2.aruco.detectMarkers(img, self.dict, parameters=self.params)
        return corners, ids

    @staticmethod
    def center(marker):
        p1, p2 = marker[0], marker[2]
        return int(p1[0] + p2[0]) // 2, int(p1[1] + p2[1]) // 2

    @staticmethod
    def dist(p1, p2):
        dx, dy = p1[0] - p2[0], p1[1] - p2[1]
        return dx, dy, math.sqrt(dx ** 2 + dy ** 2)

    def visualise(self, img, corners, ids):
        markers = len(corners)
        if markers > 0:
            y = img.shape[0] + ArucoFinder.SPAN
            img = cv2.copyMakeBorder(img, 0, (markers + 1 + 1 * (markers == 2)) * ArucoFinder.SPAN,
                                     0, 0, cv2.BORDER_CONSTANT, value=ArucoFinder.WHITE)
            cv2.aruco.drawDetectedMarkers(img, corners, ids)

            ids = np.array(ids).flatten()
            corners = np.array(corners).flatten().reshape((markers, 4, 2))
            markersDict = dict(zip(ids, corners))
            ids.sort()
            for id in ids:
                center = ArucoFinder.center(markersDict[id])
                cv2.putText(img, "Id: {}; Pos: x:{} ; y:{}".format(id, center[0], center[1]),
                            (0, y), ArucoFinder.FONT, ArucoFinder.TEXT_SCALE, ArucoFinder.BLACK)
                y += ArucoFinder.SPAN

            if markers == 2:
                c1, c2 = list(map(ArucoFinder.center, markersDict.values()))
                cv2.line(img, c1, c2, ArucoFinder.RED, 2)
                dx, dy, d = ArucoFinder.dist(c1, c2)
                cv2.putText(img, "Distance between {} and {} : {}, dx:{}, dy:{}".format(ids[0], ids[1], round(d, 3), dx, dy),
                            (0, y), ArucoFinder.FONT, ArucoFinder.TEXT_SCALE, ArucoFinder.BLACK)

        return img
