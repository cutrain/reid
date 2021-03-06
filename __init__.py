__all__ = [
    'auto_mark',
    'detect_by_frame',
    'reid_by_frame',
    'nearby',
    'reid',
    'get_data',
    'get_image',
    'detect',
    'evaluate',
    'draw_boxes',
    'cut_image',
]
import os
import cv2
import sys
import time
import pickle
import numpy as np
from tqdm import tqdm
print('init')

from .get_data import get_data
from .get_image import get_image
from .preid.identify import get_feature, evaluate
from .yolov3.detect import detect
from .util import draw_boxes, cut_image, calc_box_area_diff, calc_box_dist, TimeCounter

__module_path = os.path.dirname(sys.modules[__package__].__file__)
__dataset_path = os.path.join(__module_path, 'dataset')
if not os.path.exists(__dataset_path):
    os.mkdir(__dataset_path)


def detect_by_frame(video_capture, start_frame=0, frame_count=-1, frame_gap=0, progress=False, class_='person'):
    assert class_ in ['person'], "class {} not implemented".format(class_)
    image_iter = get_image(video_capture, frame_count=frame_count,
                               start_frame=start_frame,
                               frame_gap=frame_gap)
    for frame, image in tqdm(image_iter, disable=not progress, initial=start_frame):
        bboxes = detect(image, class_=class_)
        yield frame, image, bboxes

def reid_by_frame(video_capture, start_frame=0, frame_count=-1, frame_gap=0, progress=False, class_='person'):
    assert class_ in ['person'], "class {} not implemented".format(class_)
    for frame, image, bboxes in detect_by_frame(video_capture, frame_count=frame_count,
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

def reid_one_image(image, class_='person'):
    bboxes = detect(image, class_=class_)
    pimages = cut_image(image, bboxes)
    feature = get_feature(pimages)
    ret = {
        'feature':list(feature),
        'bboxes':bboxes,
    }
    return ret


def reid(query_path, video_path, exist_object=False,
         k=1, threshold=0.95, start_frame=0, frame_gap=0, frame_count=-1,
         progress=True, class_='person', query_optimize=True,
         auto_backup=True, backup_rate=24, save=True, load=True):
    assert class_ in ['person'], "class {} not implemented".format(class_)
    print('reid : checking files')
    assert os.path.exists(video_path), "video path is not avaliable"

    # check query type
    print('reid : reading files')
    query_images = []
    querys = []
    if not isinstance(query_path, list):
        if isinstance(query_path, str):
            image = cv2.imread(query_path)
            image = image[:,:,::-1]
            query_images.append(image)
        elif isinstance(query_path, np.ndarray):
            query_images.append(query_path)
        else:
            raise NotImplementedError
    else:
        for obj in query_path:
            if isinstance(obj, str):
                query_image = cv2.imread(obj)
                query_image = query_image[:,:,::-1]
                query_images.append(query_image)
            elif isinstance(obj, np.ndarray):
                query_images.append(obj)
            else:
                raise NotImplementedError

    print('reid : loading querys')

    # load query feature
    for query_image in query_images:
        if query_optimize:
            query_bbox = detect(query_image, class_=class_)
            if len(query_bbox) > 0:
                query_image = cut_image(query_image, [query_bbox[0]])[0]
            else:
                print("no target class object detected in query, use origin image")
        query = list(get_feature([query_image]))
        querys.extend(query)

    print('reid : loading exist data')
    # load exist data
    global __dataset_path
    exist_data = {}
    exist_len = 0
    database_path = os.path.join(__dataset_path, os.path.basename(video_path) + '.' + class_ + '.data')
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

    print('reid : prepare video')
    # prepare video
    video = get_data(video_path)
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    video_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if start_frame + frame_count < video_length and frame_count != -1:
        video_length = start_frame + frame_count

    new_data_cnt = 0
    iter_gap = 0
    eta_cnt = 10
    start = time.time()
    time_counter = [start]
    frame_counter = [start_frame]

    print('reid : start running')
    for frame_num in range(start_frame, video_length):
        # get frame info
        ret, image = video.read()
        if not ret:
            break
        image = np.flip(image, 2)
        if frame_num in exist_data:
            frame_info = exist_data[frame_num]
        else:
            frame_info = reid_one_image(image, class_=class_)
            exist_data.update({
                frame_num: frame_info
            })
            new_data_cnt += 1

        # progress
        if progress and time.time() - time_counter[-1] > 1:
            time_counter.append(time.time())
            frame_counter.append(frame_num)
            if len(time_counter) > eta_cnt:
                time_counter = time_counter[1:]
                frame_counter = frame_counter[1:]
            speed = (frame_counter[-1] - frame_counter[0]) / (time_counter[-1] - time_counter[0] + 1e-2)
            eta = (video_length - frame_num) / (speed + 1e-2)
            passed = time_counter[-1] - start
            print('\rprogress:[{}/{}] time:[{:.0f}m {:.0f}s/{:.0f}m {:.0f}s] {:.1f}iters/s'.format(
                frame_num, video_length, passed // 60, passed % 60, eta // 60, eta % 60, speed), end='')

        # query
        if len(frame_info['feature']) > 0:
            indices = evaluate(querys, frame_info['feature'], k=k, threshold=threshold)
            output = False
            for i in range(len(indices)):
                if len(indices[i]) > 0:
                    output = True
                image = draw_boxes(image, [frame_info['bboxes'][j] for j in indices[i]], copy=False)
            if output or not exist_object:
                yield image
        elif not exist_object:
            yield image

        # save data
        iter_gap += 1
        if save and auto_backup and iter_gap >= backup_rate:
            iter_gap = 0
            if new_data_cnt > 0:
                write_data = pickle.dumps(exist_data)
                with open(database_path, 'wb') as f:
                    f.write(write_data)
                with open(backup_path, 'wb') as f:
                    f.write(write_data)
                del write_data
                new_data_cnt = 0
    if save and new_data_cnt>0:
        with open(database_path, 'wb') as f:
            f.write(pickle.dumps(exist_data))
    if os.path.exists(backup_path):
        os.remove(backup_path)
    video.release()



def nearby(query_path, video_path, exist_object=False,
         k=1, threshold=0.95, start_frame=0, frame_gap=0, frame_count=-1,
         progress=True, class_='person', query_optimize=True,
         auto_backup=True, backup_rate=24, save=True, load=True,
         person_threshold=0.1, nearby_k=3):
    assert class_ in ['person'], "class {} not implemented".format(class_)
    print('checking files')
    assert os.path.exists(video_path), "video path is not avaliable"
    assert k==1

    # check query type
    query_images = []
    querys = []
    if not isinstance(query_path, list):
        if isinstance(query_path, str):
            image = cv2.imread(query_path)
            image = image[:,:,::-1]
            query_images.append(image)
        elif isinstance(query_path, np.ndarray):
            query_images.append(query_path)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    # load query feature
    for query_image in query_images:
        if query_optimize:
            query_bbox = detect(query_image, class_=class_)
            if len(query_bbox) > 0:
                query_image = cut_image(query_image, [query_bbox[0]])[0]
            else:
                print("no target class object detected in query, use origin image")
        query = list(get_feature([query_image]))
        querys.extend(query)

    # load exist data
    global __dataset_path
    exist_data = {}
    exist_len = 0
    database_path = os.path.join(__dataset_path, os.path.basename(video_path) + '.' + class_ + '.data')
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

    # prepare video
    video = get_data(video_path)
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    video_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if start_frame + frame_count < video_length and frame_count != -1:
        video_length = start_frame + frame_count

    new_data_cnt = 0
    iter_gap = 0
    eta_cnt = 10
    start = time.time()
    time_counter = [start]
    frame_counter = [start_frame]

    # -----
    nearby_person = []
    relative_dist = 100
    # -----

    for frame_num in range(start_frame, video_length):
        # get frame info
        ret, image = video.read()
        if not ret:
            break
        image = image[:,:,::-1]
        if frame_num in exist_data:
            frame_info = exist_data[frame_num]
        else:
            frame_info = reid_one_image(image, class_=class_)
            exist_data.update({
                frame_num: frame_info
            })
            new_data_cnt += 1

        # progress
        if progress and time.time() - time_counter[-1] > 1:
            time_counter.append(time.time())
            frame_counter.append(frame_num)
            if len(time_counter) > eta_cnt:
                time_counter = time_counter[1:]
                frame_counter = frame_counter[1:]
            speed = (frame_counter[-1] - frame_counter[0]) / (time_counter[-1] - time_counter[0] + 1e-2)
            eta = (video_length - frame_num) / (speed + 1e-2)
            passed = time_counter[-1] - start
            print('\rprogress:[{}/{}] time:[{:.0f}m {:.0f}s/{:.0f}m {:.0f}s] {:.1f}iters/s'.format(
                frame_num, video_length, passed // 60, passed % 60, eta // 60, eta % 60, speed), end='')

        # query
        if len(frame_info['feature']) > 0:
            indices = evaluate(querys, frame_info['feature'], k=k, threshold=threshold)[0]
            if len(indices) > 0:
                bbox = frame_info['bboxes']
                if len(bbox) > 1:
                    for i in range(len(bbox)):
                        if i == indices[0]:
                            continue
                        box_area_diff = calc_box_area_diff(bbox[i], bbox[indices[0]])
                        box_relative_dist = calc_box_dist(bbox[i], bbox[indices[0]], mode='center_relative')
                        if box_area_diff * box_relative_dist < person_threshold:
                            nearby_person.append((box_area_diff * box_relative_dist, cut_image(image, [bbox[i]])[0]))

        # save data
        iter_gap += 1
        if save and auto_backup and iter_gap >= backup_rate:
            iter_gap = 0
            if new_data_cnt > 0:
                write_data = pickle.dumps(exist_data)
                with open(database_path, 'wb') as f:
                    f.write(write_data)
                with open(backup_path, 'wb') as f:
                    f.write(write_data)
                del write_data
                new_data_cnt = 0
    if save and new_data_cnt>0:
        with open(database_path, 'wb') as f:
            f.write(pickle.dumps(exist_data))
    if os.path.exists(backup_path):
        os.remove(backup_path)
    video.release()
    nearby_person = sorted(nearby_person)[:nearby_k]
    return [i for j,i in nearby_person]


def camreid(*args, **kwargs):
    from queue import Queue
    from threading import Thread
    from .cam import webcam
    cam_size = kwargs.pop('cam_size', (640,480))
    syn = kwargs.pop('syn', False)
    cam_data_queue = Queue()
    worker = Thread(target=webcam, args=(cam_data_queue, cam_size[0], cam_size[1], syn), daemon=True)
    worker.start()
    for i in reid(*args, **kwargs):
        cam_data_queue.put(i)
    while not cam_data_queue.empty():
        time.sleep(1)

def auto_mark(path, threshold=0.99, class_='person'):
    if not os.path.exists(path):
        print('path not exists')
        raise AttributeError
    frame_count = 0
    now_id = [0]
    ret_data = {
        'video_name':path,
        'result':[],
    }
    result = ret_data['result']
    history_data = {
        'id':[],
        'feature':[],
    }

    video = get_data(path)
    video_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    time_counter = TimeCounter(video_length=video_length)

    while True:
        print(frame_count)
        ret, image = video.read()
        if not ret:
            break
        image = np.flip(image, 2)
        frame_info = reid_one_image(image, class_=class_)
        object_num = len(frame_info['feature'])
        frame_result = evaluate(frame_info['feature'], history_data['feature'], threshold=threshold)
        def get_id(x, now_id, history_data):
            if len(x)==0:
                now_id[0] += 1
                return now_id[0]
            return history_data['id'][x[0]]
        this_frame = {
            'object_num':object_num,
            'person_id':list(map(lambda x:get_id(x,now_id,history_data), frame_result)),
            'coordinate_matrix':frame_info['bboxes'],
        }
        history_data['feature'].extend(frame_info['feature'])
        history_data['id'].extend(this_frame['person_id'])
        result.append({
            'frame_index':frame_count,
            'frame_result':this_frame,
        })
        frame_count += 1
        time_counter(frame_count)
    end = time.time()
    print('proccess {} time : {}'.format(path, end-start))
    return ret_data

class auto_mark_iter:
    def __init__(self, path, class_='person', threshold=0.99):
        self._path = path
        self._video_cap = get_data(path)
        self._class_ = class_
        self._threshold = threshold

        self._len = int(self._video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._frame_count = 0
        self._history_data = {
            'feature':[],
            'id':[],
        }
        self._now_id = [0]

    def __len__(self):
        return self._len

    def __iter__(self):
        while True:
            print('now frame', self._frame_count)
            ret, image = self._video_cap.read()
            if not ret:
                break
            image = np.flip(image, 2)
            frame_info = reid_one_image(image, class_=self._class_)
            object_num = len(frame_info['feature'])
            frame_result = evaluate(frame_info['feature'], self._history_data['feature'], threshold=self._threshold)
            def get_id(x, now_id, history_data):
                if len(x) == 0:
                    now_id[0] += 1
                    return now_id[0]
                return history_data['id'][x[0]]
            this_frame = {
                'object_num':object_num,
                'coordinate_matrix':frame_info['bboxes'],
                'id':list(map(lambda x:get_id(x, self._now_id, self._history_data), frame_result)),
            }
            self._history_data['feature'].extend(frame_info['feature'])
            self._history_data['id'].extend(this_frame['id'])
            self._frame_count += 1
            yield {
                'frame_index':self._frame_count,
                'frame_result':this_frame,
            }
