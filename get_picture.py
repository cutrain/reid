import cv2
import pylab
import imageio
import skimage
import numpy as np
import pandas as pd
from .util import to_numpy

time_gap = 24
pix_gap = 3000 # 0.0038

def get_picture(video):
    global time_gap, pix_gap
    pre_time = -time_gap - 1
    fgbg = cv2.createBackgroundSubtractorKNN()
    for num, im in enumerate(video):
        fgmask = fgbg.apply(cv2.GaussianBlur(im, (5,5), 0))
        if num - pre_time > time_gap:
            temp = cv2.erode(fgmask, (3,3))
            temp = cv2.dilate(temp, (3,3))
            diff_pix = np.sum(np.sign(temp))
            if diff_pix > pix_gap:
                pre_time = num
                yield to_numpy(im)

if __name__ == "__main__":
    import get_data
    import sys
    pics = main(get_data.main(sys.argv[1], sys.argv[2]))
    for pic in pics:
        cv2.imshow('a', pic)
        k = cv2.waitKey(30) & 0xff
        if k == 29:
            break
    cv2.destroyAllWindows()

