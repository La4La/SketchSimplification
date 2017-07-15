#!/usr/bin/env python

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.datasets.image_dataset as ImageDataset
import six
import os
from PIL import Image

from chainer import cuda, optimizers, serializers, Variable
from chainer import training
from chainer.training import extensions

import argparse

import generator

from rough2lineDataset import Rough2LineDatasetNote
from training_visualizer import test_samples_simplification

#chainer.cuda.set_max_workspace_size(1024 * 1024 * 1024)
os.environ["CHAINER_TYPE_CHECK"] = "0"


def main():
    parser = argparse.ArgumentParser(
        description='chainer line drawing colorization')
    parser.add_argument('--batchsize', '-b', type=int, default=16,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=500,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset', '-i', default='./images/',
                        help='Directory of image files.')
    parser.add_argument('--out', '-o', default='./result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--snapshot_interval', type=int, default=1000,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=10,
                        help='Interval of displaying log to console')
    parser.add_argument('--test_visual_interval', type=int, default=1000,
                        help='Interval of drawing test images')
    parser.add_argument('--test_out', default='./test_result/',
                        help='DIrectory to output test samples')
    parser.add_argument('--test_image_path', default='./test_samples/test_sample3/',
                        help='Directory of image files for testing')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    root = args.dataset

    gen = generator.GEN()
    #serializers.load_npz("result_cnn/gen_iter_2000", gen)
    #print('generator loaded')

    dataset = Rough2LineDatasetNote(
        "dat/paired_dataset.dat", root + "rough/", root + "line/", root + "note", train=True,size = 328)

    train_iter = chainer.iterators.SerialIterator(dataset, args.batchsize)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        gen.to_gpu()  # Copy the model to the GPU

    # Setup optimizer parameters.
    opt = optimizers.Adam(alpha=0.0001)
    opt.setup(gen)
    opt.add_hook(chainer.optimizer.WeightDecay(1e-5), 'hook_gen')


    # Set up a trainer
    updater = cnnUpdater(
        models=(gen),
        iterator={
            'main': train_iter,
            #'test': test_iter
        },
        optimizer={
            'gen': opt,},
        device=args.gpu)

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    snapshot_interval = (args.snapshot_interval, 'iteration')
    snapshot_interval2 = (args.snapshot_interval * 2, 'iteration')
    trainer.extend(extensions.dump_graph('gen/loss'))
    trainer.extend(extensions.snapshot(), trigger=snapshot_interval2)
    trainer.extend(extensions.snapshot_object(
        gen, 'gen_iter_{.updater.iteration}'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        opt, 'optimizer_'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=(10, 'iteration'), ))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'gen/loss', 'gen/loss_L']))
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(test_samples_simplification(updater, gen, args.test_out, args.test_image_path),
                   trigger=(args.test_visual_interval, 'iteration'))

    trainer.run()

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Save the trained model
    chainer.serializers.save_npz(os.path.join(out, 'model_final'), gen)
    chainer.serializers.save_npz(os.path.join(out, 'optimizer_final'), opt)


class cnnUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen = kwargs.pop('models')
        self._iter = 0
        super(cnnUpdater, self).__init__(*args, **kwargs)

        # G_out: output of Generator
        # gt: ground truth
    def loss_gen(self, gen, G_out, gt, batchsize, alpha=1):
        xp = self.gen.xp
        loss_L = F.mean_squared_error(G_out, gt) * G_out.data.size
        loss = loss_L
        chainer.report({'loss': loss, "loss_L": loss_L}, gen)
        return loss

    def update_core(self):
        xp = self.gen.xp
        self._iter += 1

        batch = self.get_iterator('main').next()
        batchsize = len(batch)

        w_in = 328
        w_out = 328

        x_in = xp.zeros((batchsize, 1, w_in, w_in)).astype("f")
        gt = xp.zeros((batchsize, 1, w_out, w_out)).astype("f")

        for i in range(batchsize):
            x_in[i, :] = xp.asarray(batch[i][0])
            gt[i, :] = xp.asarray(batch[i][1])
        x_in = Variable(x_in)
        gt = Variable(gt)

        G_out = self.gen(x_in, test=False)

        gen_optimizer = self.get_optimizer('gen')

        gen_optimizer.update(self.loss_gen, self.gen, G_out, gt, batchsize)
        G_out.unchain_backward()


if __name__ == '__main__':
    main()

