import os
import sys
import pickle
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

__module_path = os.path.dirname(sys.modules[__package__].__file__)


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
    for frame, picture in progress_bar(picture_iter):
        bboxes = detect(picture)
        pimages = cut_image(picture, bboxes)
        for i in range(len(bboxes)):
            feature = get_feature(pimages[i])
            ret = {
                'bbox':bboxes[i],
                'image':pimages[i],
                'feature':feature,
                'frame':frame,
            }
            yield ret

def person_reid(person_paths, video_paths, k=10, progress=True):
    if progress:
        progress_bar = tqdm
    else:
        progress_bar = lambda x:x
    print('checking files')
    vpath = check_exists(video_paths)
    ppath = check_exists(person_paths)

    print('building dataset')
    dataset_feature = []
    data_images = []
    for path in vpath:
        dataset = {
            'image':[],
            'feature':[],
            'bbox':[],
            'frame':[],
        }
        database_path = os.path.join(__module_path, 'dataset', os.path.basename(path) + '.db')
        if os.path.exists(database_path):
            print('exists data')
            with open(database_path, 'rb') as f:
                dataset = pickle.loads(f.read())
        else:
            print('processing video {}'.format(path))
            video = get_data(path)
            for data in person_reid_pipeline(video, progress=True):
                dataset['feature'].append(data['feature'])
                dataset['image'].append(data['image'])
                dataset['frame'].append(data['frame'])
                dataset['bbox'].append(data['bbox'])
            with open(database_path, 'wb') as f:
                f.write(pickle.dumps(dataset))
        dataset_feature.extend(dataset['feature'])
        data_images.extend(dataset['image'])

    dataset_feature = np.stack(dataset_feature)

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
        indexs = retrieval(query, dataset_feature, k=k)
        one_ret = []
        for index in indexs:
            one_ret.append(data_images[index])
        ret.append(one_ret)
    print('finish')
    return ret


