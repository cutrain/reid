import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torchvision import transforms
import os
import sys
from PIL import Image

import scipy.io
from .re_ranking import re_ranking
from .model import ft_net, ft_net_dense

print('Loading ResNet')
__module_path = os.path.dirname(sys.modules[__package__].__file__)

batchsize = 1
which_epoch = 'last'
use_gpu = torch.cuda.is_available()

data_transforms = transforms.Compose([
        transforms.Resize((256,128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_network(network):
    save_path = os.path.join(__module_path, 'net.pth')
    network.load_state_dict(torch.load(save_path))
    return network

model_structure = ft_net(751)

model = load_network(model_structure)
model.model.fc = nn.Sequential()
model.classifier = nn.Sequential()

model = model.eval()
if use_gpu:
    model.cuda()

def dataset_loader(image):
    return data_transforms(Image.fromarray(image))

def my_dataloader(images):
    return list(map(dataset_loader, images))

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img = data
        n, c, h, w = img.size()
        count += n
        ff = torch.FloatTensor(n,2048).zero_()
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            outputs = model(input_img)
            f = outputs.data.cpu().float()
            ff = ff+f
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff), 0)
    return features

def get_feature(images, color_mode='RGB'):
    global data_transforms, model
    if len(images) == 0:
        return np.array([])
    if color_mode == 'BGR':
        images = list(map(np.flip, images, [1]*len(images)))
        images = np.flip(images, 1)
    elif color_mode == 'RGB':
        pass
    else:
        raise NotImplementedError

    dataloaders = torch.utils.data.DataLoader(my_dataloader(images), batch_size=batchsize, shuffle=False, num_workers=8)
    with torch.no_grad():
        feature = extract_feature(model, dataloaders)
    return feature.numpy()


def evaluate(query_feature, dataset_feature, k=1, threshold=0.99):
    if len(dataset_feature) == 0:
        return []
    query_feature = np.stack(query_feature)
    dataset_feature = np.stack(dataset_feature)
    q_g_dist = np.dot(query_feature, np.transpose(dataset_feature))
    ret = []
    for i in range(len(query_feature)):
        re_rank = q_g_dist[i]
        idx = np.argsort(re_rank)[-k:]
        temp = []
        for i in idx:
            if re_rank[i] > threshold:
                temp.append(i)
        temp.reverse()
        ret.append(temp)
    return ret
