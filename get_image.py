import cv2
import imageio
import skimage
import numpy as np
import pandas as pd
from .util import to_numpy

class get_image(object):
    def __init__(self, video_capture, start_frame=0, frame_gap=0):
        try:
            self.__length= int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        except Exception as e:
            self.__length = 0
        self.__reader = video_capture
        self.__start_frame = start_frame
        self.__frame_gap = frame_gap

    def __len__(self):
        return int((self.__length - self.__start_frame) / max(1, self.__frame_gap+1))

    def __iter__(self):
        pre_frame = -self.__frame_gap - 1
        if self.__length == 0:
            loop = lambda x:True
        else:
            loop = lambda x:x<self.__length
        self.__reader.set(cv2.CAP_PROP_POS_FRAMES, self.__start_frame)
        num = self.__start_frame
        while loop(num):
            ret, im = self.__reader.read()
            if num - pre_frame > self.__frame_gap:
                pre_frame = num
                yield num, np.flip(im, 2)
            num += 1

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

