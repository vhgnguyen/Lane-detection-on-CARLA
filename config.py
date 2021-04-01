IMG_WIDTH = 640 # pixel
IMG_HEIGHT = 360 # pixel

POLYFIT_MARGIN = 15 # delta of lane width fit poly

LINE_CREATE_THRESH = 100 # min pixel to create line
LINE_UPDATE_THRESH = 70 # min pixel to update line

N_HISTOGRAM = 4
N_WINDOWS = 9
WINDOW_MARGIN_RATIO = 0.08 # width of window +/- margin with ratio to img width
WINDOW_MARGIN = int(WINDOW_MARGIN_RATIO * IMG_WIDTH)
RECENTER_WINDOW_THRESH = 50 # min pixel for recenter window
POLY_SEARCH_MARGIN_RATIO = 0.1 # width of window +/- margin around previous polynom
POLY_SEARCH_MARGIN = int(POLY_SEARCH_MARGIN_RATIO * IMG_WIDTH)

YM_PER_PIX = 52 / IMG_HEIGHT
XM_PER_PIX = 3.7 / IMG_WIDTH * 0.54

