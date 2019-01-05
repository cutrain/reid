import cv2
import imageio
import skimage
import numpy as np
import pandas as pd
from .util import to_numpy

class get_image(object):
    def __init__(self, video_capture, start_frame=0, frame_count=-1, frame_gap=0):
        try:
            self.__length= int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        except Exception as e:
            self.__length = 0
        self.__reader = video_capture
        self.__new_frame = start_frame
        self.__frame_gap = frame_gap
        self.__frame_count = frame_count

    def __len__(self):
        if frame_count == -1:
            return int((self.__length - self.__now_frame) / max(1, self.__frame_gap+1))
        else:
            return frame_count

    def __iter__(self):
        pre_frame = -self.__frame_gap - 1
        if self.__length == 0:
            loop = lambda x:True
        else:
            loop = lambda x:x<self.__length
        self.__reader.set(cv2.CAP_PROP_POS_FRAMES, self.__now_frame)
        self.__now_frame
        while loop(self.__now_frame):
            ret, im = self.__reader.read()
            if self.__now_frame - pre_frame > self.__frame_gap:
                pre_frame = self.__now_frame
                yield self.__now_frame, np.flip(im, 2)
            self.__frame_count -= 1
            if self.__frame_count == 0:
                break
            self.__now_frame += 1

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

