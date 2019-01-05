import os
import cv2
import sys
import pickle
import numpy as np
from tqdm import tqdm
import imageio
imageio.plugins.ffmpeg.download()
print('init')

from .get_data import get_data
from .get_image import get_image
from .preid.identify import get_feature, evaluate
from .yolov3.detect import detect
from .util import draw_boxes, cut_image

__module_path = os.path.dirname(sys.modules[__package__].__file__)
__dataset_path = os.path.join(__module_path, 'dataset')
if not os.path.exists(__dataset_path):
    os.mkdir(__dataset_path)

__all__ = [
    'detect_by_frame',
    'reid_by_frame',
    'reid',
    'get_data',
    'get_image',
    'detect',
    'evaluate',
    'draw_boxes',
    'cut_image',
]

def detect_by_frame(video_capture, start_frame=0, frame_gap=0, progress=False, class_='person'):
    assert class_ in ['person'], "class {} not implemented".format(class_)
    image_iter = get_image(video_capture,
                               start_frame=start_frame,
                               frame_gap=frame_gap)
    for frame, image in tqdm(image_iter, disable=not progress, initial=start_frame):
        bboxes = detect(image, class_=class_)
        yield frame, image, bboxes

def reid_by_frame(video_capture, start_frame=0, frame_gap=0, progress=False, class_='person'):
    assert class_ in ['person'], "class {} not implemented".format(class_)
    for frame, image, bboxes in detect_by_frame(video_capture,
                                                  start_frame=start_frame,
                                                  frame_gap=frame_gap,
                                                  progress=progress,
                                                  class_=class_):
        ret = {
            'feature':[],
            'bboxes':bboxes,
        }
        pimages = cut_image(image, bboxes)
        feature = get_feature(pimages)
        ret['feature'].extend(list(feature))
        yield frame, image, ret

def reid(query_path, video_path, exist_object=False,
         k=None, threshold=None, start_frame=0, frame_gap=0,
         progress=True, class_='person', query_optimize=True,
         auto_backup=True, backup_rate=24, save=True, load=True):
    assert class_ in ['person'], "class {} not implemented".format(class_)
    print('checking files')
    assert os.path.exists(video_path), "video path is not avaliable"
    assert os.path.exists(query_path), "query path is not avaliable"

    # load query
    query_image = cv2.imread(query_path)
    query_image = query_image[:,:,::-1]
    if query_optimize:
        query_bbox = detect(query_image, class_=class_)
        if len(query_bbox) > 0:
            query_image = cut_image(query_image, [query_bbox[0]])[0]
        else:
            print("no target class object detected in query, use origin image")
    query = get_feature([query_image])

    # load exist data
    global __dataset_path
    exist_data = []
    exist_len = 0
    database_path = os.path.join(__dataset_path, os.path.basename(video_path) + '.realtime')
    backup_path = database_path + '.backup'
    if load and os.path.exists(database_path):
        try:
            with open(database_path, 'rb') as f:
                exist_data = pickle.loads(f.read())
        except Exception as e:
            print('data broken:', e)
            if os.path.exists(backup_path):
                print('read backup data')
                with open(backup_path, 'rb') as f:
                    exist_data = pickle.loads(f.read())
        exist_len = len(exist_data)
        print('data already exists, read {} frames'.format(exist_len))

    # query from exist data
    video = get_data(video_path)
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)

    for num, frame_info in enumerate(exist_data):
        if num < start_frame:
            continue
        ret, image = video.read()
        image = image[:,:,::-1]
        if len(frame_info['feature']) > 0:
            indices = evaluate(query, frame_info['feature'], k=k, threshold=threshold)
            image = draw_boxes(image, [frame_info['bboxes'][i] for i in indices])
            if len(indices) > 0:
                yield image
        if not exist_object:
            yield image

    # query from unprocessed data
    if exist_len < frame_count:
        print('processing {}'.format(video_path))
        if exist_len < start_frame:
            save = False
            exist_len = start_frame
        new_data = reid_by_frame(video,
                                start_frame=exist_len,
                                frame_gap=frame_gap,
                                progress=progress,
                                class_=class_)
        backup_cnt = backup_rate
        for frame, image, frame_info in new_data:
            exist_data.append(frame_info)
            backup_cnt -= 1
            if save and auto_backup and backup_cnt == 0:
                backup_cnt = backup_rate
                write_data = pickle.dumps(exist_data)
                with open(database_path, 'wb') as f:
                    f.write(write_data)
                with open(backup_path, 'wb') as f:
                    f.write(write_data)
                del write_data
            if start_frame > frame:
                print('skip', frame)
                continue
            if len(frame_info['feature']) > 0:
                indices = evaluate(query, frame_info['feature'], k=k, threshold=threshold)
                image = draw_boxes(image, [frame_info['bboxes'][i] for i in indices])
                if len(indices) > 0:
                    yield image
            if not exist_object:
                yield image
        if save:
            with open(database_path, 'wb') as f:
                f.write(pickle.dumps(exist_data))
        if os.path.exists(backup_path):
            os.remove(backup_path)
    video.release()


