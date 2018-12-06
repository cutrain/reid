import cv2
import numpy as np
import torch

from .util import to_torch

from torch.autograd import Variable
from torchvision.models import resnet50


net = resnet50(pretrained=True)
net.eval()

def get_feature(imgs):
    assert isinstance(imgs, (np.ndarray, list)), 'Type {} is not supported'.format(type(imgs))
    global net

    std_input = []
    for img in imgs:
        std_input.append(cv2.resize(img, (224, 224)))
    std_input = np.stack(std_input)
    std_input = np.transpose(std_input, (0,3,1,2))
    std_input = std_input.astype(np.float32)

    with torch.no_grad():
        std_input = to_torch(std_input)
        std_input = Variable(std_input)
        features = net(std_input)
        features = features.data.cpu().numpy()
        return features

if __name__ == "__main__":
    from scipy.misc import imread
    a = imread('./frcnn/images/1.jpeg')
    a.shape = 1,*a.shape
    b = get_feature(a)
    print(a)
