"""Custom Dataset Classes

"""
import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2 as cv
import numpy as np
import re


class YOLOAnnotationTransform(object):
    """Transforms a YOLO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_list=None):
        self.class_list = class_list

    def __call__(self, target, width, height):

        res = []
        for row in target:
            row = row.split()
            clss = float(row[0])
            bbox = row[1:5]
            x = float(bbox[0])
            y = float(bbox[1])
            w = float(bbox[2])
            h = float(bbox[3])
            bboxf = list()
            bboxf.append(x-w/2)
            bboxf.append(y-h/2)
            bboxf.append(x+w/2)
            bboxf.append(y+h/2)

            # for dim in bbox:
            #     dim = float(dim)
            #     bboxf.append(dim)

            resRow = bboxf
            resRow.append(clss)
            res.append(resRow)

        # print(res)

        return res # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class ImageTransform(object):

    def __init__(self):
        self.size = 224

    def __call__(self, image, label):

        image = np.array(image, np.float32) / 255 # scale to 0-1
        image = cv.resize(image, (self.size, self.size))
        image = torch.from_numpy(image)
        image = image.view(3, self.size, self.size)

        return image, label # [[xmin, ymin, xmax, ymax, label_ind], ... ]



class CustomDetection(data.Dataset):
    """

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
    """

    def __init__(self, root, step_size, window_size, window_sub_dir, label=False,
                 transform=ImageTransform(), target_transform=None
                 ):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self._annopath = osp.join(root, '%s.txt')
        self._imgpath = osp.join(root, '%s.jpg')
        self.name = 'Custom'
        self.step_size = step_size
        self.window_size = window_size
        self.window_sub_dir = window_sub_dir

        self.last_img_id = None

        self.window_idx = 0
        self.max_window_idx = None

        if not os.path.exists(os.path.join(root, window_sub_dir)):
            os.mkdir(os.path.join(root, window_sub_dir))


        self.ids = list()
        files = os.listdir(root)
        for file in files:
            if osp.splitext(file)[1] == '.jpg' and file.find(window_sub_dir) < 0:

                name = osp.splitext(file)[0]
                # print(name)

                if label:
                    self.ids.append(name)
                    self.split_windows(os.path.join(root, file) )
                else: 
                    window_files = os.listdir(os.path.join(root, window_sub_dir))
                    for window_file in window_files:
                        # print(window_file)
                        if window_file.find(name) >= 0 and \
                                name not in self.ids:
                            self.ids.append(name)

        print(self.ids)


    def __getitem__(self, index):
        im, l = self.pull_item(index)

        return im, l

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):

        if self.window_idx == 0:
            img_id = self.ids[index]
            self.last_img_id = img_id
            orig_path = os.path.join(self.root, img_id + '.jpg')
            orig_img = cv.imread(orig_path)
            height, width, channels = orig_img.shape
            self.max_window_idx = int(height/self.window_size *
                                    width/self.window_size)
        else:
            img_id = self.last_img_id


        label = self.get_label(img_id, self.window_idx)
        img = self.get_image(img_id, self.window_idx)


        self.window_idx += 1
        self.window_idx %= self.max_window_idx

        if self.target_transform is not None:
            height, width, channels = img.shape
            label = self.target_transform(label, width, height)

        if self.transform is not None:
            img, label = self.transform(img, label)
            # to rgb
            # img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)

        if label == 'y':
            label = [0.0, 1.0]
        else:
            label = [1.0, 0.0]


        # im = torch.from_numpy(img).permute(2, 0, 1)
        return img, torch.tensor(label)

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv.imread(self._imgpath % img_id, cv.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        # anno = ET.parse(self._annopath % img_id).getroot()
        anno = list()
        with open(self._annopath % img_id) as annoFile:
            for line in annoFile:
                anno.append(line)
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)


    def split_windows(self, img_path):
        i = j = self.window_size

        # print(img_path)
        img = cv.imread(img_path)
        height, width, channels = img.shape
        # print(height)
        # print(width)

        recs = self.get_existing_windows(img_path)
        # recs = []


        while i <= height:
            while j <= width:
                print(i)
                print(j)
                img = cv.imread(img_path)
                overlay = img.copy()

                for rec in recs:
                    if rec[2] == ord('y'):
                        color = (0, 255, 0)
                    elif rec[2] == ord('n'):
                        color = (0, 0, 255)
                    cv.rectangle(overlay, rec[0], rec[1], color, -1)
                alpha = 0.1
                cv.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

                window = img[i-self.window_size:i, j-self.window_size:j]

                cv.rectangle(img, (j-self.window_size-2,i-self.window_size-2), (j+1,i+1), \
                                (0, 0, 255), 2)

                window_path = os.path.join(self.root, self.window_sub_dir, \
                                    os.path.splitext(os.path.split(img_path)[1])[0] + \
                                    "_%d_%d" %(i, j))

                cv.imshow("Full frame", img)
                cv.imshow("Window", cv.resize(window, (self.window_size*3, self.window_size*3)))
                c = cv.waitKey()

                if c == ord('q'):
                    return
                elif c == ord(' '): # back button
                    if j > self.window_size:
                        j -= self.window_size
                        continue
                    elif i > self.window_size:
                        i -= self.window_size
                        j = width
                        continue
                    else:
                        continue
                elif c == ord('y') or c == ord('n'):
                    cv.imwrite(window_path + '.jpg', window)
                    with open(window_path + '.txt', 'w') as label_file:
                        label_file.write(chr(c))

                    recs.append([(j-self.window_size,i-self.window_size), (j,i), c])
                else:
                    continue



                j += self.window_size
            j = self.window_size
            i += self.window_size

    def get_existing_windows(self, img_path):
        
        img_id = os.path.split(img_path)[1][0:8]
        print(img_id)

        window_root = os.path.join(self.root, self.window_sub_dir)
        window_files = os.listdir(window_root)

        recs = []

        for file in window_files:
            if img_id in file and os.path.splitext(file)[1] == '.txt':
                print("found existing window: " + file)    

                with open(os.path.join(window_root, file)) as label_file:
                    c = label_file.read()      

                numbers = re.findall('(\d+)\_(\d+)\_(\d+)', file)     
                x2 = int(numbers[0][2])
                y2 = int(numbers[0][1])
                x1 = x2 - self.window_size
                y1 = y2 - self.window_size

                rec = [(x1, y1), (x2, y2), ord(c)]

                print(rec)
                recs.append(rec)

        return recs

    def id_to_window_name(self, image_id, window_idx):
        window_files = os.listdir(os.path.join(self.root, self.window_sub_dir))

        label_files = []
        to_sort = []
        # print(image_id)
        for file in window_files:
            # print(image_id)
            # print(file)
            if file.find(image_id) >= 0 and \
                    os.path.splitext(file)[1] == '.txt':
                label_files.append(file)
        
        label_files.sort()

        # print(window_idx)
        # print(label_files)
        # print(label_files[window_idx])

        return os.path.splitext(label_files[window_idx])[0]

    def get_label(self, image_id, window_idx):

        window_name = self.id_to_window_name(image_id, window_idx)
        label_file_path = os.path.join(self.root, \
                                    self.window_sub_dir, \
                                    window_name + '.txt')
        with open(label_file_path) as label_file:
            c = label_file.read()

        return c

    def get_image(self, image_id, window_idx):

        window_name = self.id_to_window_name(image_id, window_idx)
        image_file_path = os.path.join(self.root, \
                                    self.window_sub_dir, \
                                    window_name + '.jpg')

        img = cv.imread(image_file_path)

        return img





