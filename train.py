from baseNet import *

import torch
import torch.optim as optim

import os
import sys
import numpy as np

window_size = 114
batch_size = 64

num_classes = 2

size = 

net = SlidingWindowCNN(window_size, num_classes)
print(net)

# TODO try different types of opimizers and loss functions
optimizer = optim.SGD(net.parameters(), lr = 0.05)
criterion = nn.MSELoss()

params = list(net.parameters())
print(len(params))
print(params[0].size())

image_dir = input("Enter image dir: ")

files_orig = os.listdir(image_dir)
labels = []
files = []

for file in files_orig:
    if os.path.splitext(file)[1] == '.jpg':
        files.append(file)

while True:

    try:
        # 3 input channels
        input_arr = get_windows(image_dir)
        labels = get_labels(image_dir)

        optimizer.zero_grad()
        out = net(input_arr)
        target = []



    except KeyboardInterrupt:
        print("caught KeyboardInterrupt")
        break