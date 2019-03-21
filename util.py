import cv2
import math
import torch
import numpy as np

def calc_box_area_diff(box1, box2):
    width1 = width2 = height1 = height2 = None
    if len(box1) == 2:
        width1 = abs(box1[0][0] - box1[1][0])
        height1 = abs(box1[0][1] - box1[1][1])
        width2 = abs(box2[0][0] - box2[1][0])
        height2 = abs(box2[0][1] - box2[1][1])
    else:
        width1 = abs(box1[0] - box1[1])
        height1 = abs(box1[2] - box1[3])
        width2 = abs(box2[0] - box2[1])
        height2 = abs(box2[2] - box2[3])
    area1 = width1 * height1
    area2 = width2 * height2
    return (area1 - area2)**2 / (area1**2 + area2**2)

def calc_box_dist(box1, box2, mode='center'):
    assert mode in ['center', 'center_relative'], "mode {} not implemented".format(mode)
    if len(box1) == 2:
        l1 = min(box1[0][0], box1[1][0])
        r1 = max(box1[0][0], box1[1][0])
        b1 = min(box1[0][1], box1[1][1])
        t1 = max(box1[0][1], box1[1][1])
        l2 = min(box2[0][0], box2[1][0])
        r2 = max(box2[0][0], box2[1][0])
        b2 = min(box2[0][1], box2[1][1])
        t2 = max(box2[0][1], box2[1][1])
    else:
        l1 = min(box1[0], box1[1])
        r1 = max(box1[0], box1[1])
        b1 = min(box1[2], box1[3])
        t1 = max(box1[2], box1[3])
        l2 = min(box2[0], box2[1])
        r2 = max(box2[0], box2[1])
        b2 = min(box2[2], box2[3])
        t2 = max(box2[2], box2[3])
    center1_x = (l1 + r1) / 2
    center1_y = (b1 + t1) / 2
    center2_x = (l2 + r2) / 2
    center2_y = (b2 + t2) / 2
    center_dist = math.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
    if mode == 'center':
        return center_dist
    elif mode == 'center_relative':
        return center_dist / (r1-l1 + r2-l2 + 1e-3)
    else:
        raise NotImplementedError

def my_rectangle(img, p1, p2, color, thick):
    size = img.shape
    l = min(p1[0], p2[0])
    r = max(p1[0], p2[0])
    b = min(p1[1], p2[1])
    t = max(p1[1], p2[1])
    # left line
    for i in range(max(0, l-thick),min(img.shape[1],l+thick)):
        img[max(0,b-thick):min(img.shape[0],t+thick),i] = color
    # right line
    for i in range(max(0, r-thick),min(img.shape[1],r+thick)):
        img[max(0,b-thick):min(img.shape[0],t+thick),i] = color
    # bottom line
    for i in range(max(0, b-thick),min(img.shape[0],b+thick)):
        img[i,max(0,l-thick):min(img.shape[1],r+thick)] = color
    # top line
    for i in range(max(0, t-thick),min(img.shape[0],t+thick)):
        img[i,max(0,l-thick):min(img.shape[1],r+thick)] = color
    return img


def draw_boxes(img, bboxes, color=(255, 0, 0), thick=3, copy=True):
    if copy:
        imcopy = np.copy(img)
    else:
        imcopy = img
    for bbox in bboxes:
        # in cv2.rectangle, it's (col, row), (col, row) format
        if len(bbox) == 2:
            # bbox[[row1,col1],[row2,col2]]
            # cv2.rectangle(imcopy, (bbox[0][1], bbox[0][0]), (bbox[1][1], bbox[1][0]), color, thick)
            my_rectangle(imcopy, (bbox[0][1], bbox[0][0]), (bbox[1][1], bbox[1][0]), color, thick)
        else:
            # bbox[row1,row2,col1,col2]
            # cv2.rectangle(imcopy, (bbox[2], bbox[0]), (bbox[3], bbox[1]), color, thick)
            my_rectangle(imcopy, (bbox[2], bbox[0]), (bbox[3], bbox[1]), color, thick)
    return imcopy

def cut_image(img, bboxes):
    ret = []
    for bbox in bboxes:
        if len(bbox) == 2:
            # bbox[[row1,col1],[row2,col2]]
            ret.append(img[bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1]])
        else:
            # bbox[row1,row2,col1,col2]
            ret.append(img[bbox[0]:bbox[1], bbox[2]:bbox[3]])
    return ret

def to_numpy(tensor):
    import imageio
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor) == imageio.core.util.Array:
        return np.asarray(tensor)
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def hwc2cwh(arr):
    if len(arr.shape) == 3:
        return np.transpose(arr, (2,1,0))
    elif len(arr.shape) == 4:
        return np.transpose(arr, (0,3,2,1))
    else:
        assert False

def cwh2hwc(arr):
    if len(arr.shape) == 3:
        return np.transpose(arr, (2,1,0))
    elif len(arr.shape) == 4:
        return np.transpose(arr, (0,3,2,1))
    else:
        assert False

import time
class TimeCounter:
    def __init__(self, start_frame=0, video_length=0, counter_min_gap=1):
        self.video_length = video_length
        self.counter_min_gap = 1
        self.eta_cnt = 10
        self.start = time.time()
        self.time_counter = [self.start]
        self.frame_counter = [start_frame]

    def __call__(self, frame_num=0):
        if time.time() - self.time_counter[-1] > self.counter_min_gap:
            self.time_counter.append(time.time())
            self.frame_counter.append(frame_num)
            if len(self.time_counter) > self.eta_cnt:
                self.time_counter = self.time_counter[1:]
                self.frame_counter = self.frame_counter[1:]
            speed = (self.frame_counter[-1] - self.frame_counter[0]) / (self.time_counter[-1] - self.time_counter[0] + 1e-2)
            eta = (self.video_length - frame_num) / (speed + 1e-2)
            passed = self.time_counter[-1] - self.start
            print('\rprogress:[{}/{}] time:[{:.0f}m {:.0f}s/{:.0f}m {:.0f}s] {:.1f}iters/s'.format(
                frame_num, self.video_length, passed // 60, passed % 60, eta // 60, eta % 60, speed), end='')
