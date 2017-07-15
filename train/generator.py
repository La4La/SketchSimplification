#!/usr/bin/env python

import numpy as np
import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers, serializers, Variable
from chainer import function
from chainer.utils import type_check
import cupy


class GEN(chainer.Chain):

    def __init__(self):
        super(GEN, self).__init__(
            dc1=L.Convolution2D(None, 48, 5, 2, 2, wscale=0.02*math.sqrt(1*5*5)),
            fc2=L.Convolution2D(48, 128, 3, 1, 1, wscale=0.02*math.sqrt(48*3*3)),
            fc3=L.Convolution2D(128, 128, 3, 1, 1, wscale=0.02*math.sqrt(128*3*3)),
            dc4=L.Convolution2D(128, 256, 3, 2, 1, wscale=0.02*math.sqrt(128*3*3)),
            fc5=L.Convolution2D(256, 256, 3, 1, 1, wscale=0.02*math.sqrt(256*3*3)),
            fc6=L.Convolution2D(256, 256, 3, 1, 1, wscale=0.02*math.sqrt(256*3*3)),
            dc7=L.Convolution2D(256, 256, 3, 2, 1, wscale=0.02*math.sqrt(256*3*3)),
            fc8=L.Convolution2D(256, 512, 3, 1, 1, wscale=0.02*math.sqrt(256*3*3)),
            fc9=L.Convolution2D(512, 1024, 3, 1, 1, wscale=0.02*math.sqrt(512*3*3)),
            fc10=L.Convolution2D(1024, 1024, 3, 1, 1, wscale=0.02*math.sqrt(1024*3*3)),
            fc11=L.Convolution2D(1024, 1024, 3, 1, 1, wscale=0.02*math.sqrt(1024*3*3)),
            fc12=L.Convolution2D(1024, 1024, 3, 1, 1, wscale=0.02*math.sqrt(1024*3*3)),
            fc13=L.Convolution2D(1024, 512, 3, 1, 1, wscale=0.02*math.sqrt(1024*3*3)),
            fc14=L.Convolution2D(512, 256, 3, 1, 1, wscale=0.02*math.sqrt(512*3*3)),
            uc15=L.Deconvolution2D(256, 256, 4, 2, 1, wscale=0.02*math.sqrt(256*4*4)),
            fc16=L.Convolution2D(256, 256, 3, 1, 1, wscale=0.02*math.sqrt(256*3*3)),
            fc17=L.Convolution2D(256, 128, 3, 1, 1, wscale=0.02*math.sqrt(256*3*3)),
            uc18=L.Deconvolution2D(128, 128, 4, 2, 1, wscale=0.02*math.sqrt(128*4*4)),
            fc19=L.Convolution2D(128, 128, 3, 1, 1, wscale=0.02*math.sqrt(128*3*3)),
            fc20=L.Convolution2D(128, 48, 3, 1, 1, wscale=0.02*math.sqrt(128*3*3)),
            uc21=L.Deconvolution2D(48, 48, 4, 2, 1, wscale=0.02*math.sqrt(48*4*4)),
            fc22=L.Convolution2D(48, 24, 3, 1, 1, wscale=0.02*math.sqrt(48*3*3)),
            fc23=L.Convolution2D(24, 1, 3, 1, 1, wscale=0.02*math.sqrt(24*3*3)),

            bn1=L.BatchNormalization(48),
            bn2=L.BatchNormalization(128),
            bn3=L.BatchNormalization(128),
            bn4=L.BatchNormalization(256),
            bn5=L.BatchNormalization(256),
            bn6=L.BatchNormalization(256),
            bn7=L.BatchNormalization(256),
            bn8=L.BatchNormalization(512),
            bn9=L.BatchNormalization(1024),
            bn10=L.BatchNormalization(1024),
            bn11=L.BatchNormalization(1024),
            bn12=L.BatchNormalization(1024),
            bn13=L.BatchNormalization(512),
            bn14=L.BatchNormalization(256),
            bn15=L.BatchNormalization(256),
            bn16=L.BatchNormalization(256),
            bn17=L.BatchNormalization(128),
            bn18=L.BatchNormalization(128),
            bn19=L.BatchNormalization(128),
            bn20=L.BatchNormalization(48),
            bn21=L.BatchNormalization(48),
            bn22=L.BatchNormalization(24)
        )

    def __call__(self, x, test=False, use_cudnn=False):
        h = F.relu(self.bn1(self.dc1(x), test=test), use_cudnn)
        h = F.relu(self.bn2(self.fc2(h), test=test), use_cudnn)
        h = F.relu(self.bn3(self.fc3(h), test=test), use_cudnn)
        h = F.relu(self.bn4(self.dc4(h), test=test), use_cudnn)
        h = F.relu(self.bn5(self.fc5(h), test=test), use_cudnn)
        h = F.relu(self.bn6(self.fc6(h), test=test), use_cudnn)
        h = F.relu(self.bn7(self.dc7(h), test=test), use_cudnn)
        h = F.relu(self.bn8(self.fc8(h), test=test), use_cudnn)
        h = F.relu(self.bn9(self.fc9(h), test=test), use_cudnn)
        h = F.relu(self.bn10(self.fc10(h), test=test), use_cudnn)
        h = F.relu(self.bn11(self.fc11(h), test=test), use_cudnn)
        h = F.relu(self.bn12(self.fc12(h), test=test), use_cudnn)
        h = F.relu(self.bn13(self.fc13(h), test=test), use_cudnn)
        h = F.relu(self.bn14(self.fc14(h), test=test), use_cudnn)
        h = F.relu(self.bn15(self.uc15(h), test=test), use_cudnn)
        h = F.relu(self.bn16(self.fc16(h), test=test), use_cudnn)
        h = F.relu(self.bn17(self.fc17(h), test=test), use_cudnn)
        h = F.relu(self.bn18(self.uc18(h), test=test), use_cudnn)
        h = F.relu(self.bn19(self.fc19(h), test=test), use_cudnn)
        h = F.relu(self.bn20(self.fc20(h), test=test), use_cudnn)
        h = F.relu(self.bn21(self.uc21(h), test=test), use_cudnn)
        h = F.relu(self.bn22(self.fc22(h), test=test), use_cudnn)
        h = F.relu(self.fc23(h), use_cudnn)
        #h = Variable(cupy.where(h.data < 0.9, 0, h.data))

        return h


class DilatedGEN(chainer.Chain):

    def __init__(self):
        super(DilatedGEN, self).__init__(
            dc1=L.Convolution2D(None, 48, 5, 2, 2, wscale=0.02*math.sqrt(1*5*5)),
            fc2=L.Convolution2D(48, 128, 3, 1, 1, wscale=0.02*math.sqrt(48*3*3)),
            fc3=L.Convolution2D(128, 128, 3, 1, 1, wscale=0.02*math.sqrt(128*3*3)),
            dc4=L.Convolution2D(128, 256, 3, 2, 1, wscale=0.02*math.sqrt(128*3*3)),
            fc5=L.Convolution2D(256, 256, 3, 1, 1, wscale=0.02*math.sqrt(256*3*3)),
            fc6=L.Convolution2D(256, 256, 3, 1, 1, wscale=0.02*math.sqrt(256*3*3)),
            dc7=L.Convolution2D(256, 256, 3, 2, 1, wscale=0.02*math.sqrt(256*3*3)),
            fc8=L.Convolution2D(256, 512, 3, 1, 1, wscale=0.02*math.sqrt(256*3*3)),
            fc9=L.DilatedConvolution2D(512, 1024, 3, 1, 2, dilate=2, wscale=0.02*math.sqrt(512*3*3)),
            fc10=L.DilatedConvolution2D(1024, 1024, 3, 1, 4, dilate=4, wscale=0.02*math.sqrt(1024*3*3)),
            fc11=L.DilatedConvolution2D(1024, 1024, 3, 1, 8, dilate=8, wscale=0.02*math.sqrt(1024*3*3)),
            fc12=L.Convolution2D(1024, 1024, 3, 1, 1, wscale=0.02*math.sqrt(1024*3*3)),
            fc13=L.Convolution2D(1024, 512, 3, 1, 1, wscale=0.02*math.sqrt(1024*3*3)),
            fc14=L.Convolution2D(512, 256, 3, 1, 1, wscale=0.02*math.sqrt(512*3*3)),
            uc15=L.Deconvolution2D(256, 256, 4, 2, 1, wscale=0.02*math.sqrt(256*4*4)),
            fc16=L.Convolution2D(256, 256, 3, 1, 1, wscale=0.02*math.sqrt(256*3*3)),
            fc17=L.Convolution2D(256, 128, 3, 1, 1, wscale=0.02*math.sqrt(256*3*3)),
            uc18=L.Deconvolution2D(128, 128, 4, 2, 1, wscale=0.02*math.sqrt(128*4*4)),
            fc19=L.Convolution2D(128, 128, 3, 1, 1, wscale=0.02*math.sqrt(128*3*3)),
            fc20=L.Convolution2D(128, 48, 3, 1, 1, wscale=0.02*math.sqrt(128*3*3)),
            uc21=L.Deconvolution2D(48, 48, 4, 2, 1, wscale=0.02*math.sqrt(48*4*4)),
            fc22=L.Convolution2D(48, 24, 3, 1, 1, wscale=0.02*math.sqrt(48*3*3)),
            fc23=L.Convolution2D(24, 1, 3, 1, 1, wscale=0.02*math.sqrt(24*3*3)),

            bn1=L.BatchNormalization(48),
            bn2=L.BatchNormalization(128),
            bn3=L.BatchNormalization(128),
            bn4=L.BatchNormalization(256),
            bn5=L.BatchNormalization(256),
            bn6=L.BatchNormalization(256),
            bn7=L.BatchNormalization(256),
            bn8=L.BatchNormalization(512),
            bn9=L.BatchNormalization(1024),
            bn10=L.BatchNormalization(1024),
            bn11=L.BatchNormalization(1024),
            bn12=L.BatchNormalization(1024),
            bn13=L.BatchNormalization(512),
            bn14=L.BatchNormalization(256),
            bn15=L.BatchNormalization(256),
            bn16=L.BatchNormalization(256),
            bn17=L.BatchNormalization(128),
            bn18=L.BatchNormalization(128),
            bn19=L.BatchNormalization(128),
            bn20=L.BatchNormalization(48),
            bn21=L.BatchNormalization(48),
            bn22=L.BatchNormalization(24)
        )

    def __call__(self, x, test=False, use_cudnn=False):
        h = F.relu(self.bn1(self.dc1(x), test=test), use_cudnn)
        h = F.relu(self.bn2(self.fc2(h), test=test), use_cudnn)
        h = F.relu(self.bn3(self.fc3(h), test=test), use_cudnn)
        h = F.relu(self.bn4(self.dc4(h), test=test), use_cudnn)
        h = F.relu(self.bn5(self.fc5(h), test=test), use_cudnn)
        h = F.relu(self.bn6(self.fc6(h), test=test), use_cudnn)
        h = F.relu(self.bn7(self.dc7(h), test=test), use_cudnn)
        h = F.relu(self.bn8(self.fc8(h), test=test), use_cudnn)
        h = F.relu(self.bn9(self.fc9(h), test=test), use_cudnn)
        h = F.relu(self.bn10(self.fc10(h), test=test), use_cudnn)
        h = F.relu(self.bn11(self.fc11(h), test=test), use_cudnn)
        h = F.relu(self.bn12(self.fc12(h), test=test), use_cudnn)
        h = F.relu(self.bn13(self.fc13(h), test=test), use_cudnn)
        h = F.relu(self.bn14(self.fc14(h), test=test), use_cudnn)
        h = F.relu(self.bn15(self.uc15(h), test=test), use_cudnn)
        h = F.relu(self.bn16(self.fc16(h), test=test), use_cudnn)
        h = F.relu(self.bn17(self.fc17(h), test=test), use_cudnn)
        h = F.relu(self.bn18(self.uc18(h), test=test), use_cudnn)
        h = F.relu(self.bn19(self.fc19(h), test=test), use_cudnn)
        h = F.relu(self.bn20(self.fc20(h), test=test), use_cudnn)
        h = F.relu(self.bn21(self.uc21(h), test=test), use_cudnn)
        h = F.relu(self.bn22(self.fc22(h), test=test), use_cudnn)
        h = F.relu(self.fc23(h), use_cudnn)
        #h = Variable(cupy.where(h.data < 0.9, 0, h.data))

        return h

class mGEN(chainer.Chain):
    def __init__(self):
        super(mGEN, self).__init__(
            dc1=L.Convolution2D(None, 48, 5, 2, 2, wscale=0.02 * math.sqrt(1 * 5 * 5)),
            fc2=L.Convolution2D(48, 128, 3, 1, 1, wscale=0.02 * math.sqrt(48 * 3 * 3)),
            fc3=L.Convolution2D(128, 128, 3, 1, 1, wscale=0.02 * math.sqrt(128 * 3 * 3)),
            dc4=L.Convolution2D(128, 256, 3, 2, 1, wscale=0.02 * math.sqrt(128 * 3 * 3)),
            fc5=L.Convolution2D(256, 256, 3, 1, 1, wscale=0.02 * math.sqrt(256 * 3 * 3)),
            fc6=L.Convolution2D(256, 256, 3, 1, 1, wscale=0.02 * math.sqrt(256 * 3 * 3)),
            dc7=L.Convolution2D(256, 256, 3, 2, 1, wscale=0.02 * math.sqrt(256 * 3 * 3)),
            fc8=L.Convolution2D(256, 512, 3, 1, 1, wscale=0.02 * math.sqrt(256 * 3 * 3)),
            fc9=L.DilatedConvolution2D(512, 1024, 3, 1, 2, dilate=2, wscale=0.02 * math.sqrt(512 * 3 * 3)),
            fc10=L.DilatedConvolution2D(1024, 1024, 3, 1, 4, dilate=4, wscale=0.02 * math.sqrt(1024 * 3 * 3)),
            fc11=L.DilatedConvolution2D(1024, 1024, 3, 1, 8, dilate=8, wscale=0.02 * math.sqrt(1024 * 3 * 3)),
            fc12=L.Convolution2D(1024, 1024, 3, 1, 1, wscale=0.02 * math.sqrt(1024 * 3 * 3)),
            fc13=L.Convolution2D(1024, 512, 3, 1, 1, wscale=0.02 * math.sqrt(1024 * 3 * 3)),
            fc14=L.Convolution2D(512, 256, 3, 1, 1, wscale=0.02 * math.sqrt(512 * 3 * 3)),
            uc15=L.Deconvolution2D(256, 256, 4, 2, 1, wscale=0.02 * math.sqrt(256 * 4 * 4)),
            fc16=L.Convolution2D(256, 256, 3, 1, 1, wscale=0.02 * math.sqrt(256 * 3 * 3)),
            fc17=L.Convolution2D(256, 128, 3, 1, 1, wscale=0.02 * math.sqrt(256 * 3 * 3)),
            uc18=L.Deconvolution2D(128, 128, 4, 2, 1, wscale=0.02 * math.sqrt(128 * 4 * 4)),
            fc19=L.Convolution2D(128, 128, 3, 1, 1, wscale=0.02 * math.sqrt(128 * 3 * 3)),
            fc20=L.Convolution2D(128, 48, 3, 1, 1, wscale=0.02 * math.sqrt(128 * 3 * 3)),
            uc21=L.Deconvolution2D(48, 48, 4, 2, 1, wscale=0.02 * math.sqrt(48 * 4 * 4)),
            fc22=L.Convolution2D(48, 24, 3, 1, 1, wscale=0.02 * math.sqrt(48 * 3 * 3)),
            fc23=L.Convolution2D(25, 1, 3, 1, 1, wscale=0.02 * math.sqrt(24 * 3 * 3)),

            bn1=L.BatchNormalization(48),
            bn2=L.BatchNormalization(128),
            bn3=L.BatchNormalization(128),
            bn4=L.BatchNormalization(256),
            bn5=L.BatchNormalization(256),
            bn6=L.BatchNormalization(256),
            bn7=L.BatchNormalization(256),
            bn8=L.BatchNormalization(512),
            bn9=L.BatchNormalization(1024),
            bn10=L.BatchNormalization(1024),
            bn11=L.BatchNormalization(1024),
            bn12=L.BatchNormalization(1024),
            bn13=L.BatchNormalization(512),
            bn14=L.BatchNormalization(256),
            bn15=L.BatchNormalization(256),
            bn16=L.BatchNormalization(256),
            bn17=L.BatchNormalization(128),
            bn18=L.BatchNormalization(128),
            bn19=L.BatchNormalization(128),
            bn20=L.BatchNormalization(48),
            bn21=L.BatchNormalization(48),
            bn22=L.BatchNormalization(24)
        )

    def __call__(self, x, m, test=False, use_cudnn=False):
        h = F.relu(self.bn1(self.dc1(x), test=test), use_cudnn)
        h = F.relu(self.bn2(self.fc2(h), test=test), use_cudnn)
        h = F.relu(self.bn3(self.fc3(h), test=test), use_cudnn)
        h = F.relu(self.bn4(self.dc4(h), test=test), use_cudnn)
        h = F.relu(self.bn5(self.fc5(h), test=test), use_cudnn)
        h = F.relu(self.bn6(self.fc6(h), test=test), use_cudnn)
        h = F.relu(self.bn7(self.dc7(h), test=test), use_cudnn)
        h = F.relu(self.bn8(self.fc8(h), test=test), use_cudnn)
        h = F.relu(self.bn9(self.fc9(h), test=test), use_cudnn)
        h = F.relu(self.bn10(self.fc10(h), test=test), use_cudnn)
        h = F.relu(self.bn11(self.fc11(h), test=test), use_cudnn)
        h = F.relu(self.bn12(self.fc12(h), test=test), use_cudnn)
        h = F.relu(self.bn13(self.fc13(h), test=test), use_cudnn)
        h = F.relu(self.bn14(self.fc14(h), test=test), use_cudnn)
        h = F.relu(self.bn15(self.uc15(h), test=test), use_cudnn)
        h = F.relu(self.bn16(self.fc16(h), test=test), use_cudnn)
        h = F.relu(self.bn17(self.fc17(h), test=test), use_cudnn)
        h = F.relu(self.bn18(self.uc18(h), test=test), use_cudnn)
        h = F.relu(self.bn19(self.fc19(h), test=test), use_cudnn)
        h = F.relu(self.bn20(self.fc20(h), test=test), use_cudnn)
        h = F.relu(self.bn21(self.uc21(h), test=test), use_cudnn)
        h = F.relu(self.bn22(self.fc22(h), test=test), use_cudnn)
        h = F.relu(self.fc23(F.concat([h, m])), use_cudnn)
        # h = Variable(cupy.where(h.data < 0.9, 0, h.data))

        return h

