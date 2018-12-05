from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dset
from scipy.misc import imread
from .lib.model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from .lib.model.rpn.bbox_transform import clip_boxes
from .lib.model.nms.nms_wrapper import nms
from .lib.model.rpn.bbox_transform import bbox_transform_inv
from .lib.model.faster_rcnn.vgg16 import vgg16
from .lib.model.faster_rcnn.resnet import resnet


def __get_real_input(im, caffe=False):
    pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= pixel_means

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    MAX_SIZE = 1000
    target_size = 600

    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > MAX_SIZE:
        im_scale = float(MAX_SIZE) / float(im_size_max)

    blob = []
    for img in im_orig:
        blob.append(cv2.resize(img, None, None, fx=im_scale, fy=im_scale,
                               interpolation=cv2.INTER_LINEAR))

    return np.stack(blob), np.full(len(im), im_scale)


def extract_boxes(dets, threshold):
    select = np.where(dets[:,-1] > threshold)
    return np.round(dets[select][:,:4]).astype(np.int)[:,(1,3,0,2)]



def detect(imgs, class_name='person'):
    """
    param: ndarray (batch_size, w, h, c) or (batch_size, w, h)  with RGB channel
    """
    global fasterRCNN, im_data, im_info, num_boxes, gt_boxes, class_col, pascal_classes, cuda
    assert class_name in ['person'], "{} is not supported class".format(class_name)
    num_images = len(imgs)
    print('load {} images'.format(num_images))

    if len(imgs.shape) == 3:
        imgs = imgs[:,:,:,np.newaxis]
        imgs = np.concatenate((imgs, imgs, imgs), axis=3)
    blob, im_scales = __get_real_input(imgs)
    im_info_np = np.array([
        [blob.shape[1], blob.shape[2], im_scales[0]] * num_images
    ], dtype=np.float32)

    im_data_pt = torch.from_numpy(blob)
    im_data_pt = im_data_pt.permute(0, 3, 1, 2)
    im_info_pt = torch.from_numpy(im_info_np)

    im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
    im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
    gt_boxes.data.resize_(num_images, 1, 5).zero_()
    num_boxes.data.resize_(1).zero_()

    print('data prepare end')

    rois, cls_prob, bbox_pred, \
    rpn_loss_cls, rpn_loss_box, \
    RCNN_loss_cls, RCNN_loss_bbox, \
    rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

    print('predict end')

    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]

    # Apply bounding-box regression deltas
    box_deltas = bbox_pred.data
    # Optionally normalize targets by a precomputed mean and stdev
    if cuda:
        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                   + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
    else:
        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                   + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
    box_deltas = box_deltas.view(num_images, -1, 4 * len(pascal_classes))

    pred_boxes = bbox_transform_inv(boxes, box_deltas, num_images)
    pred_boxes = clip_boxes(pred_boxes, im_info.data, num_images)

    pred_boxes /= im_scales[0]

    print('normalize end')

    # TODO: check 1 size dim
    # scores = scores.squeeze()
    # pred_boxes = pred_boxes.squeeze()

    vis = False
    thresh = 0.05
    max_per_image = 100
    NMS = 0.3

    col = class_col[class_name]
    bboxes = []

    for i in range(num_images):
        inds = torch.nonzero(scores[i][:,col]>thresh).view(-1)
        # if there is det
        if inds.numel() > 0:
            cls_scores = scores[i][:,col][inds]
            _, order = torch.sort(cls_scores, 0, True)
            cls_boxes = pred_boxes[i][inds][:, col * 4:(col+1) * 4]
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            cls_dets = cls_dets[order]
            keep = nms(cls_dets, NMS, force_cpu= not cuda)
            cls_dets = cls_dets[keep.view(-1).long()]
            bboxes.append(extract_boxes(cls_dets.cpu().numpy(), 0.8))
        else:
            bboxes.append([])

    print('bbox check end')
    return bboxes





#bbox save people bbox information
boxs = [[]]

# load model before start
module_path = os.path.dirname(sys.modules[__package__].__file__)
model_path = os.path.join(module_path, './model/faster_rcnn.pth')
if not os.path.exists(model_path):
    raise Exception("There is no model, maybe you should create a directory 'model' and put a model named 'resnet101_caffe.pth' in it")

# load config file
config_path = os.path.join(module_path, './cfgs/vgg16.yml')
cfg_from_file(config_path)


pascal_classes = np.asarray(['__background__', 'person'])
class_col = {
    'person':1,
}

fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=False)
fasterRCNN.create_architecture()

cuda = True if torch.cuda.is_available() else False
cfg.USE_GPU_NMS = cuda

print('start load model')

if cuda:
    checkpoint = torch.load(model_path)
else:
    checkpoint = torch.load(model_path, map_location=(lambda storage, loc: storage))
fasterRCNN.load_state_dict(checkpoint['model'])
print('load model successfully!')

im_data = torch.FloatTensor(1)
im_info = torch.FloatTensor(1)
num_boxes = torch.LongTensor(1)
gt_boxes = torch.FloatTensor(1)

if cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()


im_data = Variable(im_data, volatile=True)
im_info = Variable(im_info, volatile=True)
num_boxes = Variable(num_boxes, volatile=True)
gt_boxes = Variable(gt_boxes, volatile=True)

if cuda:
    fasterRCNN.cuda()

fasterRCNN.eval()
