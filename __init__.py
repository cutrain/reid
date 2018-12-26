import os
import numpy as np
from tqdm import tqdm
import imageio
imageio.plugins.ffmpeg.download()
print('init')

from .get_data import get_data
from .get_picture import get_picture
from .get_feature import get_feature
from .frcnn import detect
from .retrieval import retrieval
from .util import draw_boxes, cut_image


def check_exists(paths):
    exist_path = []
    for path in paths:
        if os.path.exists(path):
            exist_path.append(path)
        else:
            print('{} not found, skipped'.format(path))
    return exist_path

def person_reid_pipeline(video_capture, frame_gap=24, progress=False):
    if progress:
        progress_bar = tqdm
    else:
        progress_bar = lambda x:x
    picture_iter = get_picture(video_capture, frame_gap=frame_gap)
    for picture in progress_bar(picture_iter):
        bboxes = detect(picture)
        pimages = cut_image(picture, bboxes)
        for pimage in pimages:
            feature = get_feature(pimage)
            yield pimage, feature

def person_reid(person_paths, video_paths, k=10, progress=True):
    if progress:
        progress_bar = tqdm
    else:
        progress_bar = lambda x:x
    print('checking files')
    vpath = check_exists(video_paths)
    ppath = check_exists(person_paths)

    print('building dataset')
    dataset = []
    data_images = []
    for path in vpath:
        print('processing video {}'.format(path))
        video = get_data(path)
        for pimage, feature in person_reid_pipeline(video, progress=True):
            data_images.append(pimage)
            dataset.append(feature)
    dataset = np.stack(dataset)

    print('finish building dataset')

    print('building query ... ')

    features = []
    for path in progress_bar(ppath):
        from scipy.misc import imread
        picture = imread(path)
        feature = get_feature(picture)
        features.append(feature)
    querys = features
    print('finish building query')

    print('retrievaling')
    ret = []
    for query in querys:
        indexs = retrieval(query, dataset, k=k)
        one_ret = []
        for index in indexs:
            one_ret.append(data_images[index])
        ret.append(one_ret)
    print('finish')
    return ret


