import os
import numpy as np
import imageio
imageio.plugins.ffmpeg.download()

from .get_data import get_data
from .get_picture import get_picture
from .get_feature import get_feature
from .vehicle_detection import detect_car
from .frcnn import detect
from .retrieval import retrieval
from .util import draw_boxes, cut_image


def sample():
    dataset_path = './a.mp4'
    query_path = './a.jpg'
    video = get_data(path)
    pictures = get_picture(video)
    dataset = []
    for picture in pictures:
        bboxes = detect_car(picture)
        cars = cut_image(picture, bboxes)
        features = get_feature(cars)
        dataset.append(features)
    import numpy as np
    dataset = np.stack(dataset)
    query = imageio.read(query_path)
    near_pictures = retrieval(query, dataset, k=10)

def check_exists(paths):
    exist_path = []
    for path in paths:
        if os.path.exists(path):
            exist_path.append(path)
        else:
            print('{} not found, skipped'.format(path))
    return exist_path

def person_reid(person_paths, video_paths, k=10):
    print('checking files')
    vpath = check_exists(video_paths)
    ppath = check_exists(person_paths)

    print('building dataset')
    dataset = []
    cut_images = []
    for path in vpath:
        print('get video {} ... '.format(path), end='')
        video = get_data(path)
        print('extract pictures ... ', end='')
        pictures = list(get_picture(video))
        print('detect person ... ', end='')
        bboxes = detect(pictures)
        for i in range(len(bboxes)):
            cut_images.extend(cut_image(pictures[i], bboxes[i]))
        print('get feature ... ', end='')
        features = get_feature(cut_images)
        dataset.append(features)
        print('finish video {}'.format(path))
    dataset = np.stack(dataset)

    print('finish building dataset')

    print('building query ... ', end='')

    pictures = []
    for path in ppath:
        from scipy.misc import imread
        picture = imread(path)
        pictures.append(picture)
    print('detect person ... ', end='')
    bboxes = detect(pictures)
    cut_images = []
    for i in range(len(bboxes)):
        cut_images.extend(cut_image(pictures[i], bboxes[i]))
    print('get feature ... ', end='')
    features = get_feature(cut_images)
    querys = features
    print('finish building query')

    print('retrievaling')
    ret = []
    for query in querys:
        indexs = retrieval(query, dataset, k=k)
        one_ret = []
        for index in indexs:
            one_ret.append(cut_images[index])
        ret.append(one_ret)
    print('finish')
    return ret


