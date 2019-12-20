import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from torchvision import models

import os




class SlidingWindowCNN(nn.Module):


    def __init__(self, size, num_classes):
        super(SlidingWindowCNN, self).__init__()
        self.size = size

        # TODO: try vgg19 or other variants 
        self.vgg = models.vgg16(pretrained=False).features
        # self.slice1 = torch.nn.Sequential()
        # self.slice2 = torch.nn.Sequential()
        # self.slice3 = torch.nn.Sequential()
        # self.slice4 = torch.nn.Sequential()
        # self.slice5 = torch.nn.Sequential()

        # for x in range(4):
        #     self.slice1.add_module(str(x), vgg[x])
        # for x in range(4, 9):
        #     self.slice2.add_module(str(x), vgg[x])
        # for x in range(9, 16):
        #     self.slice3.add_module(str(x), vgg[x])
        # for x in range(16, 23):
        #     self.slice4.add_module(str(x), vgg[x])
        # for x in range(23, 27)


        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )


    def forward(self, x):
        # h = self.slice1(x)
        # h_relu1_2 = h 
        # h = self.slice2(h)
        # h_relu2_2 = h 
        # h = self.slice3(h)
        # h_relu3_3 = h 
        # h = self.slice4(h)
        # h_relu4_3 = h 

        # vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        # out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        
        for k in range(len(self.vgg)):
            x = self.vgg[k](x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)


        return x

        # return out 