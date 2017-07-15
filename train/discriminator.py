#!/usr/bin/env python

import numpy as np
import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers, serializers, Variable
from chainer import function
from chainer.utils import type_check


class DIS(chainer.Chain):
    def __init__(self):
        super(DIS, self).__init__(
            c1=L.Convolution2D(1, 16, 5, 2, 2, wscale=0.02*math.sqrt(5*5*1)),
            c2=L.Convolution2D(16, 32, 3, 2, 1, wscale=0.02*math.sqrt(3*3*16)),
            c3=L.Convolution2D(32, 64, 3, 2, 1, wscale=0.02*math.sqrt(3*3*32)),
            c4=L.Convolution2D(64, 128, 3, 2, 1, wscale=0.02*math.sqrt(3*3*64)),
            c5=L.Convolution2D(128, 256, 3, 2, 1, wscale=0.02*math.sqrt(3*3*128)),
            c6=L.Convolution2D(256, 512, 3, 2, 1, wscale=0.02*math.sqrt(3*3*256)),
            c7=L.Linear(4*4*512, 2, wscale=0.02*math.sqrt(4*4*512)),

            bn1=L.BatchNormalization(16),
            bn2=L.BatchNormalization(32),
            bn3=L.BatchNormalization(64),
            bn4=L.BatchNormalization(128),
            bn5=L.BatchNormalization(256),
            bn6=L.BatchNormalization(512)
        )

    def __call__(self, x, test=False):
        h = F.relu(self.bn1(self.c1(x), test=test))
        h = F.relu(self.bn2(self.c2(h), test=test))
        h = F.relu(self.bn3(self.c3(h), test=test))
        h = F.relu(self.bn4(self.c4(h), test=test))
        h = F.dropout(F.relu(self.bn5(self.c5(h), test=test)), train=not test, ratio=0.5)
        h = F.dropout(F.relu(self.bn6(self.c6(h), test=test)), train=not test, ratio=0.5)
        h = self.c7(h)

        return h