from .util import to_torch

from torch.autograd import Variable
from torchvision.models import resnet50


net = resnet50(pretrained=True)
net.eval()

def get_feature(pic):
    global net
    pic = to_torch(pic)
    pic = Variable(pic, volatile=True)
    return net(pic).data.cpu()

if __name__ == "__main__":
    get_feature()
