import cv2
import pylab
import imageio
import skimage
import numpy as np
import pandas as pd

time_gap = 24
pix_gap = 3000 # 0.0038

def main(video):
    global time_gap, pix_gap
    pre_time = -time_gap - 1
    fgbg = cv2.createBackgroundSubtractorKNN()
    for num, im in enumerate(video):
        if num - pre_time > time_gap:
            im = cv2.GaussianBlur(im, (5,5), 1)
            fgmask = fgbg.apply(im)
            diff_pix = np.sum(np.sign(fgmask))
            if diff_pix > pix_gap:
                pre_time = num
                yield im

if __name__ == "__main__":
    import get_data
    import sys
    pics = main(get_data.main(sys.argv[1], sys.argv[2]))
    cnt = 0
    pre = None
    for pic in pics:
        cnt += 1
        pylab.imshow(pic)
        pylab.show()
        pre = pic

