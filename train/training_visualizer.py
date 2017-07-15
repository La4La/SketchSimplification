import os
import glob
import numpy as np

import chainer
import chainer.cuda
from chainer import cuda, serializers, Variable
import chainer.functions as F
import cv2

def test_samples_simplification(updater, generator, output_path, test_image_path, s_size=324, use_noise=False):
    @chainer.training.make_extension()
    def read_img(path):
        image1 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if image1.shape[0] < image1.shape[1]:
            s0 = s_size
            s1 = int(image1.shape[1] * (s_size / image1.shape[0]))
            s1 = s1 - s1 % 16
        else:
            s1 = s_size
            s0 = int(image1.shape[0] * (s_size / image1.shape[1]))
            s0 = s0 - s0 % 16

        image1 = np.asarray(image1, np.float32)
        image1 = cv2.resize(image1, (s1, s0), interpolation=cv2.INTER_AREA)

        if image1.ndim == 2:
            image1 = image1[:, :, np.newaxis]
        #print(image1.shape)

        if use_noise and image1.shape[1] != s_size:
            return image1.transpose(2, 1, 0), True
        return image1.transpose(2, 0, 1), False

    def save_as_img(array, name, transposed=False):
        if transposed:
            array = array.transpose(2, 1, 0)
        else:
            array = array.transpose(1, 2, 0)

        array = array * 255
        array = array.clip(0, 255).astype(np.uint8)
        img = cuda.to_cpu(array)

        cv2.imwrite(name, img)

    def simplify(file_path, output_path):
        sample, transposed = read_img(file_path)
        #print("sample shape")
        #print(sample.shape)
        #print(transposed)
        x_in = np.zeros((1, 1, sample.shape[1], sample.shape[2]), dtype='f')
        x_in[0, :] = sample[0]
        x_in = cuda.to_gpu(x_in)
        cnn_in = Variable(x_in)

        if use_noise:
            z_in = np.zeros((1, 1, sample.shape[1], sample.shape[2]), dtype='f')
            z_rnd = np.random.normal(size=(s_size)).astype("f")
            #print(z_rnd.shape)
            z_reshape = np.ones((1, sample.shape[1], s_size)).astype("f")
            #print(z_reshape.shape)
            z_in[0]=z_rnd*z_reshape
            z_in = cuda.to_gpu(z_in)
            z_in = Variable(z_in)
            cnn_in = F.concat([cnn_in, z_in, z_in, z_in])

        cnn_out = generator(cnn_in, test=True)

        save_as_img(cnn_out.data[0], output_path, transposed)

    def samples_simplify(trainer):
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        file_test = glob.glob(test_image_path+'*.jpg')
        for f in file_test:
            filename = os.path.basename(f)
            filename = os.path.splitext(filename)[0]
            simplify(f, output_path+"/iter_"+str(trainer.updater.iteration)+"_"+filename+".jpg")

    return samples_simplify