import cv2
import torch
import imageio
import numpy as np

def draw_boxes(img, bboxes, color=(255, 0, 0), thick=3):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        if len(bbox) == 2:
            cv2.rectangle(imcopy, (bbox[0][1], bbox[0][0]), (bbox[1][1], bbox[1][0]), color, thick)
        else:
            cv2.rectangle(imcopy, (bbox[2], bbox[0]), (bbox[3], bbox[1]), color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def cut_image(img, bboxes):
    ret = []
    for bbox in bboxes:
        if len(bbox) == 2:
            ret.append(img[bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1]])
        else:
            ret.append(img[bbox[0]:bbox[1], bbox[2]:bbox[3]])
    return ret

def to_numpy(tensor):
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

