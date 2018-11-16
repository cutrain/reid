import cv2
import numpy as np
import matplotlib.image as mpimg

def pipeline_svm(img):
    from .svm_pipeline import vehicle_detection_svm
    return vehicle_detection_svm(img, None, None)

def detect_car(path_or_img, model='svm'):
    assert model in ['svm'], "model %s not support now" % model
    img = path_or_img
    if type(path_or_img) == str:
        img = mpimg.imread(path_or_img)

    origin_size = img.shape[:2]
    img = cv2.resize(img, (1280, 720))

    if model == 'svm':
        bboxes = pipeline_svm(img)
    else:
        assert False

    prate = (origin_size[1] / 1280, origin_size[0] / 720)
    real_bboxes = []
    for bbox in bboxes:
        real_bboxes.append(((int(bbox[0][0]*prate[0]), int(bbox[0][1]*prate[1])), (int(bbox[1][0]*prate[0]), int(bbox[1][1]*prate[1]))))


    return real_bboxes


