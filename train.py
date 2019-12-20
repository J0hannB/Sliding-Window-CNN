from baseNet import *
from data_utils import*

import torch
import torch.optim as optim
import torch.utils.data as data

import os
import sys
import numpy as np

window_size = 114
batch_size = 32

num_classes = 2

iter_count = 0

if len(sys.argv) < 2:
    print("must specify images directory")
    sys.exit()

image_dir = sys.argv[1]


net = SlidingWindowCNN(window_size, num_classes)
print(net)

# TODO try different types of opimizers and loss functions
optimizer = optim.SGD(net.parameters(), lr = 0.0001)
criterion = nn.MSELoss()

params = list(net.parameters())
print(len(params))
print(params[0].size())

# image_dir = input("Enter image dir: ")

dataset = CustomDetection(image_dir, window_size, window_size, 'windows', label=False) #True)


data_loader = data.DataLoader(dataset, batch_size, 
                                num_workers=0, 
                                shuffle=True,
                                pin_memory=False)

batch_iterator = iter(data_loader)

# files_orig = os.listdir(image_dir)
# labels = []
# files = []

# for file in files_orig:
#     if os.path.splitext(file)[1] == '.jpg':
#         files.append(file)

while True:
    try:

        iter_count ++ 1

        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        optimizer.zero_grad()
        out = net(images)
        loss = criterion(out, targets)
        print("Loss: " + str(loss.data))
        loss.backward()
        optimizer.step() # Does the update

        if iter_count % 100 == 0:
            torch.save(net.state_dict(), "./saved_model_sliding_window.pth")




    except KeyboardInterrupt:
        print("caught KeyboardInterrupt")
        break