import imageio
imageio.plugins.ffmpeg.download()

from .get_data import get_data
from .get_picture import get_picture
from .get_feature import get_feature
from .vehicle_detection import detect_car
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


