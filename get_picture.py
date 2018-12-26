import cv2
import imageio
import skimage
import numpy as np
import pandas as pd
from .util import to_numpy

class get_picture(object):
    def __init__(self, video_capture, frame_gap=24, pix_gap=3000):
        try:
            self.__length= int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        except Exception as e:
            self.__length = 0
        self.__reader = video_capture
        self.__frame_gap = frame_gap
        self.__pix_gap = pix_gap

    def __len__(self):
        return int(self.__length / max(1, self.__frame_gap+1))
    def __iter__(self):
        pre_frame = -self.__frame_gap - 1
        # fgbg = cv2.createBackgroundSubtractorKNN()
        for num in range(self.__length):
            ret, im = self.__reader.read()
            # fgmask = fgbg.apply(cv2.GaussianBlur(im, (5,5), 0))
            if num - pre_frame > self.__frame_gap:
                pre_frame = num
                yield np.flip(im, 2)
                # temp = cv2.erode(fgmask, (3,3))
                # temp = cv2.dilate(temp, (3,3))
                # diff_pix = np.sum(np.sign(temp))
                # if diff_pix > pix_gap:
                    # pre_frame = num
                    # yield np.flip(im, 2)

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

