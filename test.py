from baseNet import *
from data_utils import*

import torch
import torch.optim as optim
import torch.utils.data as data

import os
import sys
import numpy as np
import argparse

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='Train sliding window CNN')
parser.add_argument('--saved_model',
                    default=None, type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='./', type=str,
                    help='File path to save results')
parser.add_argument('--cuda', default=False, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--images_path', default='./Images',
                    help='Location of images root directory')
parser.add_argument('--window_size', default=114, type=int,
                    help='size of sliding window')

args = parser.parse_args()
print(args)

# if torch.cuda.is_available() and args.cuda:
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')
# else:
#     torch.set_default_tensor_type('torch.FloatTensor')


num_classes = 2


net = SlidingWindowCNN(args.window_size, num_classes)
print(net)

if args.cuda:
    net = torch.nn.DataParallel(net)

if args.saved_model is None:
    print("must specify saved model")
    sys.exit()

print('Resuming training, loading {}...'.format(args.resume))
net.load_weights(args.resume)

if args.cuda:
    net = net.cuda()

params = list(net.parameters())
print(len(params))
print(params[0].size())


dataset = CustomDetection(args.images_path, args.window_size, args.window_size, 'windows', label=False) #True)


data_loader = data.DataLoader(dataset, args.batch_size, 
                                num_workers=0, 
                                shuffle=True,
                                pin_memory=True)

for i in range(len(data_loader.ids)):
    num_windows = ### How do I get this?
    windows = []
    window_gts= []
    window_classifications = []
    for j in num_windows:
        im, gt = data_loader.pull_item(i)
        
        # hndle cuda?

        out = net(im.unsqueeze(0))

        





while True:
    try:

        iter_count += 1

        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        if args.cuda:
            images = images.cuda()
            targets = targets.cuda()

        optimizer.zero_grad()
        out = net(images)
        loss = criterion(out, targets)
        print("Iter {}, Loss: ".format(iter_count) + str(loss.data))
        loss.backward()
        optimizer.step() # Does the update

        if iter_count % 100 == 0:
            print("saving model after {} iterations".format(iter_count))
            save_path = os.path.join(args.save_folder, "saved_model_sliding_window_{}.pth".format(iter_count))
            torch.save(net.state_dict(), save_path)




    except KeyboardInterrupt:
        print("caught KeyboardInterrupt")
        break