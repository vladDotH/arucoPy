import cv2
from cv2 import aruco

# Настройка доски ChAruco
CHROW = 7  # ширина (в клетках)
CHCOL = 5  # высота (в клетках)
CHSQUARE_LEN = 0.037  # длина клетки в метрах
CHMARKER_LEN = 0.022  # длина маркера в метрах (на доске)
CHSQUARES = (CHCOL - 1) * (CHROW - 1)
CHARUCO_DICT = aruco.Dictionary_get(aruco.DICT_6X6_1000)  # Словарь маркеров на доске
CHPARAMS = aruco.DetectorParameters_create()
#
MARKER_LEN = 0.026  # длина маркера в метрах (при работе)

CAM_PORT = 1
CALIB_FILE = 'calib.pckl'

ESC = 27
SPACE = 32
TIME_DELAY = 10

SPAN = 30
FONT = cv2.FONT_HERSHEY_SIMPLEX
BLACK = (0, 0, 0)
WHITE = (0xff, 0xff, 0xff)
RED = (0, 0, 0xff)
TEXT_SCALE = 0.5
PRECISION = 3
