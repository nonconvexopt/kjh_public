import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.datasets as dsets
import torch.utils as utils
import torchvision.transforms as transforms
from torch.nn.modules.utils import _pair

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


trainset = dsets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
trainloader = utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)

testset = dsets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)


class MVN_Conv2d_out:
    def __init__(self, in_channel, out_channel, kernel_size):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.kernel_size = _pair(kernel_size)
        self.dist = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(self.kernel_size[0] * self.kernel_size[1]),
            torch.eye(self.kernel_size[0] * self.kernel_size[1])
        )

    def sample(self):
        return self.dist.sample().unsqueeze(0).unsqueeze(0).repeat(self.in_channel, self.out_channel, 1, 1)

class Conv2d_VI(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(nn.Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        self.std_normal = MVN_Conv2d_out(in_channels, out_channels, kernel_size)
        self.var_filter = nn.Parameter(
            torch.Tensor((in_channels, out_channels, kernel_size[0], kernel_size[1])).normal_(0, 1), requires_grad = True)
        self.var_bias = nn.Parameter(torch.Tensor((in_channels, out_channels)).normal_(0, 1))


    def sample(self):
        params = self.std_normal.sample()
        params = params * self.var_filter
        params += self.mu_filter
        conv2d = nn.Conv2d(in_channels = self.in_channels, out_channels = self.out_channels, kernel_size = self.kernel_size,
                           stride = self.stride, padding = self.padding, dilation = self.dilation, groups = self.groups,
                           bias = self.bias, padding_mode = self.padding_mode)
        conv2d.weight.data = params
        return conv2d

    def forward(self):
        return self.sample()

class Bayesian_VGG(nn.Module):
    def __init__(self):
        super(Bayesian_VGG, self).__init__()
        self.params = nn.Sequential(
            Conv2d_VI(3, 64, (3, 3), stride=(1, 1), padding=1),
            Conv2d_VI(64, 64, (3, 3), stride=1, padding=1),
            nn.MaxPool2d((2, 2), stride=2, padding=0),

            Conv2d_VI(64, 128, (3, 3), stride=1, padding=1),
            Conv2d_VI(128, 128, (3, 3), stride=1, padding=1),
            nn.MaxPool2d((2, 2), stride=2, padding=0),

            Conv2d_VI(128, 256, (3, 3), stride=1, padding=1),
            Conv2d_VI(256, 256, (3, 3), stride=1, padding=1),
            Conv2d_VI(256, 256, (3, 3), stride=1, padding=1),
            nn.MaxPool2d((2, 2), stride=2, padding=0),

            Conv2d_VI(256, 512, (3, 3), stride=1, padding=1),
            Conv2d_VI(512, 512, (3, 3), stride=1, padding=1),
            Conv2d_VI(512, 512, (3, 3), stride=1, padding=1),
            nn.MaxPool2d((2, 2), stride=2, padding=0),

            Conv2d_VI(512, 512, (3, 3), stride=1, padding=1),
            Conv2d_VI(512, 512, (3, 3), stride=1, padding=1),
            Conv2d_VI(512, 512, (3, 3), stride=1, padding=1),
            nn.MaxPool2d((2, 2), stride=2, padding=0),

            nn.Linear(512, 10)
        )



    def forward(self, x):
        return self.params(x)


def loss(pred, label):
    return 1



model = Bayesian_VGG()
for x, y in trainloader:
    print(model(x))
    break


