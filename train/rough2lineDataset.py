#!/usr/bin/env python

import numpy as np
import chainer
'''
import chainer.functions as F
import chainer.links as L
import chainer.datasets.image_dataset as ImageDataset
'''
import six
import os
import glob

from chainer import cuda, optimizers, serializers, Variable
import cv2


class Rough2LineDataset(chainer.dataset.DatasetMixin):

    def __init__(self, paths, root1='./input', root2='./target', dtype=np.float32,
                 leak=(0, 0), size = 384, root_ref = None, train=False, input_norm=False):
        if isinstance(paths, six.string_types):
            with open(paths) as paths_file:
                paths = [path.strip() for path in paths_file]
        self._paths = paths
        self._root1 = root1
        self._root2 = root2
        self._root_ref = root_ref
        self._dtype = dtype
        self._leak = leak
        self._size = size
        self._img_dict = {}
        self._train = train
        self._input_norm = input_norm

    def set_img_dict(self, img_dict):
        self._img_dict = img_dict

    def get_vec(self, name):
        tag_size = 1539
        v = np.zeros(tag_size).astype(np.int32)
        if name in self._img_dict.keys():
            for i in self._img_dict[name][3]:
                v[i] = 1
        return v

    def __len__(self):
        return len(self._paths)

    def get_name(self, i):
        return self._paths[i]

    def get_example(self, i, minimize=False, log=False, bin_r=0):
        if self._train:
            bin_r = 0.9

        readed = False
        if np.random.rand() < bin_r:
            if np.random.rand() < 0.3:
                path1 = os.path.join(self._root1 + "_rough/", self._paths[i])
            else:
                path1 = os.path.join(self._root1 + "_line/", self._paths[i])
            path2 = os.path.join(self._root2 + "_rough/", self._paths[i])
            image1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
            image2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
            if image1 is not None and image2 is not None:
                if image1.shape[0] > 0 and image1.shape[1] and image2.shape[0] > 0 and image2.shape[1]:
                    readed = True
        if not readed:
            path1 = os.path.join(self._root1, self._paths[i])
            path2 = os.path.join(self._root2, self._paths[i])
            image1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
            image2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

        # input image size: 384*384

        # randomly down sampling
        if self._train:
            scale = np.random.choice(range(6,15)) / 6.0
            row = int(image1.shape[0] // scale)
            col = int(image1.shape[1] // scale)
            if row >= self._size and col >= self._size:
                image1 = cv2.resize(image1, (row, col))
                image2 = cv2.resize(image2, (row, col))
            elif row <= col:
                image1 = cv2.resize(image1, (self._size, int(col / row * self._size)))
                image2 = cv2.resize(image2, (self._size, int(col / row * self._size)))
            elif row > col:
                image1 = cv2.resize(image1, (int(row / col * self._size), self._size))
                image2 = cv2.resize(image2, (int(row / col * self._size), self._size))

        # randomly crop
        if self._train:
            #print(image1.shape)
            x = np.random.randint(0, image1.shape[1] - self._size + 1)
            y = np.random.randint(0, image1.shape[0] - self._size + 1)
            image1 = image1[y:y+self._size, x:x+self._size]
            image2 = image2[y:y+self._size, x:x+self._size]

        # add flip
        if self._train:
            if np.random.rand() > 0.5:
                image1 = cv2.flip(image1, 1)
                image2 = cv2.flip(image2, 1)
            if np.random.rand() > 0.9:
                image1 = cv2.flip(image1, 0)
                image2 = cv2.flip(image2, 0)

        # replace rough sketches with line arts
        if self._train:
            if np.random.rand() > 0.9:
                image1 = image2

        if self._input_norm:
            image1 = np.asarray(image1/255.0, self._dtype)
        else:
            image1 = np.asarray(image1, self._dtype)
        image2 = np.asarray(image2/255.0, self._dtype)
        image2 = np.where(image2<0.9, 0, image2)

        # image is grayscale
        if image1.ndim == 2:
            image1 = image1[:, :, np.newaxis]
        if image2.ndim == 2:
            image2 = image2[:, :, np.newaxis]

        image1 = (image1.transpose(2, 0, 1))
        image2 = (image2.transpose(2, 0, 1))

        return image1, image2

class Rough2LineDatasetwithMASK(chainer.dataset.DatasetMixin):

    def __init__(self, paths, root1='./input', root2='./target', dtype=np.float32,
                 leak=(0, 0), size = 384, root_ref = None, train=False, mask = 3, input_norm = True):
        if isinstance(paths, six.string_types):
            with open(paths) as paths_file:
                paths = [path.strip() for path in paths_file]
        self._paths = paths
        self._root1 = root1
        self._root2 = root2
        self._root_ref = root_ref
        self._dtype = dtype
        self._leak = leak
        self._size = size
        self._img_dict = {}
        self._train = train
        self._mask = mask
        self._input_norm = input_norm

    def set_img_dict(self, img_dict):
        self._img_dict = img_dict

    def get_vec(self, name):
        tag_size = 1539
        v = np.zeros(tag_size).astype(np.int32)
        if name in self._img_dict.keys():
            for i in self._img_dict[name][3]:
                v[i] = 1
        return v

    def __len__(self):
        return len(self._paths)

    def get_name(self, i):
        return self._paths[i]

    def get_example(self, i, minimize=False, log=False, bin_r=0):
        if self._train:
            bin_r = 0.9

        readed = False
        if np.random.rand() < bin_r:
            if np.random.rand() < 0.3:
                path1 = os.path.join(self._root1 + "_rough/", self._paths[i])
            else:
                path1 = os.path.join(self._root1 + "_line/", self._paths[i])
            path2 = os.path.join(self._root2 + "_rough/", self._paths[i])
            image1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
            image2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
            if image1 is not None and image2 is not None:
                if image1.shape[0] > 0 and image1.shape[1] and image2.shape[0] > 0 and image2.shape[1]:
                    readed = True
        if not readed:
            path1 = os.path.join(self._root1, self._paths[i])
            path2 = os.path.join(self._root2, self._paths[i])
            image1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
            image2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

        # input image size: 384*384

        # randomly down sampling
        if self._train:
            scale = np.random.choice(range(6,15)) / 6.0
            row = int(image1.shape[0] // scale)
            col = int(image1.shape[1] // scale)
            if row >= self._size and col >= self._size:
                image1 = cv2.resize(image1, (row, col))
                image2 = cv2.resize(image2, (row, col))
            elif row <= col:
                image1 = cv2.resize(image1, (self._size, int(col / row * self._size)))
                image2 = cv2.resize(image2, (self._size, int(col / row * self._size)))
            elif row > col:
                image1 = cv2.resize(image1, (int(row / col * self._size), self._size))
                image2 = cv2.resize(image2, (int(row / col * self._size), self._size))

        # randomly crop
        if self._train:
            #print(image1.shape)
            x = np.random.randint(0, image1.shape[1] - self._size + 1)
            y = np.random.randint(0, image1.shape[0] - self._size + 1)
            image1 = image1[y:y+self._size, x:x+self._size]
            image2 = image2[y:y+self._size, x:x+self._size]

        # add flip
        if self._train:
            if np.random.rand() > 0.5:
                image1 = cv2.flip(image1, 1)
                image2 = cv2.flip(image2, 1)
            if np.random.rand() > 0.9:
                image1 = cv2.flip(image1, 0)
                image2 = cv2.flip(image2, 0)

        # replace rough sketches with line arts
        if self._train:
            if np.random.rand() > 0.9:
                image1 = image2

        if self._input_norm:
            image1 = np.asarray(image1/255.0, self._dtype)
        else:
            image1 = np.asarray(image1, self._dtype)
        image2 = np.asarray(image2/255.0, self._dtype)
        image2 = np.where(image2<0.9, 0, image2)

        # image is grayscale
        if image1.ndim == 2:
            image1 = image1[:, :, np.newaxis]
        if image2.ndim == 2:
            image2 = image2[:, :, np.newaxis]

        # generate mask
        image1 = np.insert(image1, 1, 1, axis=2)
        mask = np.ones((self._size, self._size, 1), dtype = self._dtype)
        if self._train:
            iterNum = np.random.randint(5, 15)
            #print('--' + str(iterNum))
            maskNum = 0
            for i in range(iterNum):
                x = np.random.randint(0, self._size - 2)
                y = np.random.randint(0, self._size - 2)
                w = np.random.randint(2, 15)
                h = np.random.randint(2, 15)
                sum2 = np.sum(image2[x:min(x+w, self._size-1), y:min(y+h, self._size-1), 0])
                #print(str(sum2) + ' ' + str(w*h*1.0*0.75))
                if sum2 < w*h*1.0 * 0.75:
                    maskNum += 1
                    image1[x:min(x+w, self._size-1), y:min(y+h, self._size-1), 1] \
                        = image1[x:min(x+w, self._size-1), y:min(y+h, self._size-1), 0]
                    mask[x:min(x+w, self._size-1), y:min(y+h, self._size-1), 0] = self._mask
            #print(maskNum)

        image1 = (image1.transpose(2, 0, 1))
        image2 = (image2.transpose(2, 0, 1))

        return image1, image2, mask


class Rough2LineDatasetNote(chainer.dataset.DatasetMixin):

    def __init__(self, paths, root1='./input', root2='./target', note='./note', dtype=np.float32,
                 leak=(0, 0), size = 384, root_ref = None, train=False, input_norm=False):
        if isinstance(paths, six.string_types):
            with open(paths) as paths_file:
                paths = [path.strip() for path in paths_file]
        self._paths = paths
        self._root1 = root1
        self._root2 = root2
        note = glob.glob(note + '/*.jpg')
        self._note = note
        self._root_ref = root_ref
        self._dtype = dtype
        self._leak = leak
        self._size = size
        self._img_dict = {}
        self._train = train
        self._input_norm = input_norm

    def set_img_dict(self, img_dict):
        self._img_dict = img_dict

    def get_vec(self, name):
        tag_size = 1539
        v = np.zeros(tag_size).astype(np.int32)
        if name in self._img_dict.keys():
            for i in self._img_dict[name][3]:
                v[i] = 1
        return v

    def __len__(self):
        return len(self._paths)

    def get_name(self, i):
        return self._paths[i]

    def get_example(self, i, minimize=False, log=False, bin_r=0):
        if self._train:
            bin_r = 0.9

        readed = False
        if np.random.rand() < bin_r:
            if np.random.rand() < 0.3:
                path1 = os.path.join(self._root1 + "_rough/", self._paths[i])
            else:
                path1 = os.path.join(self._root1 + "_line/", self._paths[i])
            path2 = os.path.join(self._root2 + "_rough/", self._paths[i])
            image1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
            image2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
            if image1 is not None and image2 is not None:
                if image1.shape[0] > 0 and image1.shape[1] and image2.shape[0] > 0 and image2.shape[1]:
                    readed = True
        if not readed:
            path1 = os.path.join(self._root1, self._paths[i])
            path2 = os.path.join(self._root2, self._paths[i])
            image1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
            image2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

        # input image size: 384*384

        # randomly down sampling
        if self._train:
            scale = np.random.choice(range(6,15)) / 6.0
            row = int(image1.shape[0] // scale)
            col = int(image1.shape[1] // scale)
            if row >= self._size and col >= self._size:
                image1 = cv2.resize(image1, (row, col))
                image2 = cv2.resize(image2, (row, col))
            elif row <= col:
                image1 = cv2.resize(image1, (self._size, int(col / row * self._size)))
                image2 = cv2.resize(image2, (self._size, int(col / row * self._size)))
            elif row > col:
                image1 = cv2.resize(image1, (int(row / col * self._size), self._size))
                image2 = cv2.resize(image2, (int(row / col * self._size), self._size))

        # randomly crop
        if self._train:
            #print(image1.shape)
            x = np.random.randint(0, image1.shape[1] - self._size + 1)
            y = np.random.randint(0, image1.shape[0] - self._size + 1)
            image1 = image1[y:y+self._size, x:x+self._size]
            image2 = image2[y:y+self._size, x:x+self._size]

        # add flip
        if self._train:
            if np.random.rand() > 0.5:
                image1 = cv2.flip(image1, 1)
                image2 = cv2.flip(image2, 1)
            if np.random.rand() > 0.9:
                image1 = cv2.flip(image1, 0)
                image2 = cv2.flip(image2, 0)

        # replace rough sketches with line arts
        if self._train:
            if np.random.rand() > 0.9:
                image1 = image2

        # add lines of note
        if self._train and np.random.rand() > 0.9:
            path3 = np.random.choice(self._note)
            noteImg = cv2.imread(path3, cv2.IMREAD_GRAYSCALE)

            # random down sampling
            scale = np.random.choice(range(6, 30)) / 6.0
            row = int(noteImg.shape[0] // scale)
            col = int(noteImg.shape[1] // scale)
            if row >= self._size and col >= self._size:
                noteImg = cv2.resize(noteImg, (row, col))
            elif row <= col:
                noteImg = cv2.resize(noteImg, (self._size, int(col / row * self._size)))
            elif row > col:
                noteImg = cv2.resize(noteImg, (int(row / col * self._size), self._size))

            # random rotation
            noteImg = 255 - noteImg
            alpha = np.random.randint(2) * 90
            beta = np.random.normal() * 5
            row_n, col_n = noteImg.shape
            M = cv2.getRotationMatrix2D((self._size / 2, self._size / 2), alpha + beta, 1)
            noteImg = cv2.warpAffine(noteImg, M, (col_n, row_n))
            noteImg = 255 - noteImg

            # random crop
            x = np.random.randint(0, noteImg.shape[1] - self._size + 1)
            y = np.random.randint(0, noteImg.shape[0] - self._size + 1)
            noteImg = noteImg[y:y + self._size, x:x + self._size]

            lineColor = np.random.randint(0, 200)
            noteImg = np.where(noteImg < 245, lineColor, noteImg)

            image1 = np.where(noteImg < 200, noteImg, image1)

        if self._input_norm:
            image1 = np.asarray(image1/255.0, self._dtype)
        else:
            image1 = np.asarray(image1, self._dtype)
        image2 = np.asarray(image2/255.0, self._dtype)
        image2 = np.where(image2<0.9, 0, image2)

        # image is grayscale
        if image1.ndim == 2:
            image1 = image1[:, :, np.newaxis]
        if image2.ndim == 2:
            image2 = image2[:, :, np.newaxis]

        image1 = (image1.transpose(2, 0, 1))
        image2 = (image2.transpose(2, 0, 1))

        return image1, image2