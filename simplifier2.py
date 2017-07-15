import os
import glob
import numpy as np
import argparse

from chainer import cuda, serializers, Variable
import cv2
import generator

parser = argparse.ArgumentParser(description='')
parser.add_argument('--model1', default='/home/gao/lineart/result_model/simplification_model2')
parser.add_argument('--model2', default='/home/gao/lineart/result_model/simplification_model1')
parser.add_argument('--output', default='./simplification_output2')
parser.add_argument('--test_path', default='./test_samples/test_sample5')
parser.add_argument('--size', type=int, default=648)
parser.add_argument('--gpu', '-g', type=int, default=-1)
parser.add_argument('--concat', default=False)
parser.add_argument('--mode', type=int, default=1)
args = parser.parse_args()

def read_img(path, s_size):
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

    return image1.transpose(2, 0, 1), False

def save_as_img(array, name, origin, transposed=False):
    if transposed:
        origin = origin.transpose(2, 1, 0)
        array = array.transpose(2, 1, 0)
    else:
        origin = origin.transpose(1, 2, 0)
        array = array.transpose(1, 2, 0)

    array = array * 255
    array = array.clip(0, 255).astype(np.uint8)
    img = cuda.to_cpu(array)
    origin = origin.clip(0, 255).astype(np.uint8)

    if args.concat:
        img_concat = cv2.hconcat([origin, img])
        cv2.imwrite(name, img_concat)
    else:
        cv2.imwrite(name, img)

def simplify(file_path, simplifier1, simplifier2, output_path, s_size):
    origin, transposed = read_img(file_path, s_size)
    x_in = np.zeros((1, 1, origin.shape[1], origin.shape[2]), dtype='f')
    x_in[0, :] = origin[0]
    x_in = cuda.to_gpu(x_in)
    cnn_in = Variable(x_in)
    cnn_out = simplifier1(cnn_in, test=True)

    if args.mode == 1:
        cnn_in = (cnn_out.data * 255).clip(0,255).astype(np.float32)
        cnn_out = simplifier2(cnn_in, test=True)

    save_as_img(cnn_out.data[0], output_path, origin, transposed)


if __name__ == '__main__':
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    simplifier1 = generator.GEN()
    serializers.load_npz(args.model1, simplifier1)
    simplifier2 = generator.GEN()
    serializers.load_npz(args.model2, simplifier2)
    print('simplifier loaded')

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        simplifier1.to_gpu()
        simplifier2.to_gpu()

    file_test = glob.glob(args.test_path+'/*.jpg')
    for f in file_test:
        filename = os.path.basename(f)
        print(filename)
        filename = os.path.splitext(filename)[0]
        simplify(f, simplifier1, simplifier2, args.output+"/"+filename+".jpg", args.size)