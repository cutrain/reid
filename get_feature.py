import cv2
import numpy as np
import torch

from .util import to_torch

from torch.autograd import Variable
from torchvision.models import resnet50


def get_feature(img):
    assert isinstance(img, np.ndarray), 'Type {} is not supported'.format(type(img))
    global net

    std_input = [cv2.resize(img, (224, 224))]
    std_input = np.stack(std_input)
    std_input = np.transpose(std_input, (0,3,1,2))
    std_input = std_input.astype(np.float32)

    with torch.no_grad():
        std_input = to_torch(std_input)
        if torch.cuda.is_available():
            std_input = std_input.cuda()
        features = net(std_input)
        features = features.data.cpu().numpy()
        return features[0]

print('init resnet50')
net = resnet50(pretrained=True)
if torch.cuda.is_available():
    net.cuda()
net.eval()

