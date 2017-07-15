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
import discriminator

from rough2lineDataset import Rough2LineDataset, Rough2LineDatasetNote
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
    parser.add_argument('--test_visual_interval', type=int, default=500,
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
    serializers.load_npz("/mnt/sakura201/gao/SketchSimplification/result_model1/gen_iter_8000", gen)
    print('generator loaded')

    dis = discriminator.DIS()
    serializers.load_npz("/mnt/sakura201/gao/SketchSimplification/result_model1/gen_dis_iter_8000", dis)
    print('discriminator loaded')

    # paired dataset
    paired_dataset = Rough2LineDatasetNote(
        "dat/paired_dataset.dat", root + "rough/", root + "line/", root + "note",
        train=True, size = 256)
    paired_iter = chainer.iterators.MultiprocessIterator(paired_dataset, args.batchsize)

    # unpaired dataset
    unpaired_dataset = Rough2LineDataset(
        "dat/unpaired_dataset.dat", root + "unpaired_rough/", root + "unpaired_line/",
        train=True, size = 256)
    unpaired_iter = chainer.iterators.MultiprocessIterator(unpaired_dataset, args.batchsize)


    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        gen.to_gpu()  # Copy the model to the GPU
        dis.to_gpu()  # Copy the model to the GPU

    # Setup optimizer parameters.
    opt = optimizers.Adam(alpha=0.0001)
    opt.setup(gen)
    opt.add_hook(chainer.optimizer.WeightDecay(1e-5), 'hook_gen')

    opt_d = chainer.optimizers.Adam(alpha=0.0001)
    opt_d.setup(dis)
    opt_d.add_hook(chainer.optimizer.WeightDecay(1e-5), 'hook_dec')

    # Set up a trainer
    updater = ganUpdater(
        models=(gen, dis),
        iterator={
            'main': paired_iter,
            'unpaired': unpaired_iter
            #'test': test_iter
        },
        optimizer={
            'gen': opt,
            'dis': opt_d},
        device=args.gpu)

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    #serializers.load_npz('/mnt/sakura201/gao/SketchSimplification/result_model/snapshot_iter_8000', trainer)

    snapshot_interval = (args.snapshot_interval, 'iteration')
    snapshot_interval2 = (args.snapshot_interval * 2, 'iteration')
    trainer.extend(extensions.dump_graph('gen/loss'))
    trainer.extend(extensions.snapshot(), trigger=snapshot_interval2)
    trainer.extend(extensions.snapshot_object(
        gen, 'gen_iter_{.updater.iteration}'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dis, 'gen_dis_iter_{.updater.iteration}'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        opt, 'optimizer_'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=(10, 'iteration'), ))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'gen/loss', 'gen/loss_L', 'gen/loss_adv', 'gen/loss_adv_u', 'dis/loss', 'dis/fake_p', 'dis/real_p', 'dis/fake_u', 'dis/real_u']))
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


class ganUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        self._iter = 0
        super(ganUpdater, self).__init__(*args, **kwargs)

    def line_loss(self, x, t, k=3):
        #print(x.data.type)
        lx = x - F.max_pooling_2d(x, k, 1, 1)
        lt = t - F.max_pooling_2d(t, k, 1, 1)
        return 2 * F.mean_absolute_error(lx, lt)

        # 0 for dataset
        # 1 for fake
        # G_p_rough: output of Generator (paired rough sketch)
        # p_line: paired line art
    def loss_gen(self, gen, G_p_rough, D_p_rough, p_line, D_u_rough, batchsize, alpha=0.1, beta=0.1):
        xp = self.gen.xp
        loss_L = F.mean_squared_error(G_p_rough, p_line) * G_p_rough.data.shape[0]
        loss_adv = F.softmax_cross_entropy(D_p_rough, Variable(xp.zeros(batchsize, dtype=np.int32)))
        loss_adv_unpaired = F.softmax_cross_entropy(D_u_rough, Variable(xp.zeros(batchsize, dtype=np.int32)))
        #loss_line = self.line_loss(G_p_rough, p_line)
        loss = loss_L + alpha * loss_adv + beta * loss_adv_unpaired #+ loss_line
        chainer.report({'loss': loss, "loss_L": loss_L, 'loss_adv': loss_adv, 'loss_adv_u': loss_adv_unpaired}, gen)
        return loss

    def loss_dis(self, dis, D_p_rough, p_line, D_u_rough, u_line, batchsize, alpha=0.1, beta=0.1):
        xp = self.gen.xp
        loss_fake_p = F.softmax_cross_entropy(D_p_rough, Variable(xp.ones(batchsize, dtype=np.int32)))
        loss_real_p = F.softmax_cross_entropy(self.dis(p_line), Variable(xp.zeros(batchsize, dtype=np.int32)))
        loss_fake_u = F.softmax_cross_entropy(D_u_rough, Variable(xp.ones(batchsize, dtype=np.int32)))
        loss_real_u = F.softmax_cross_entropy(self.dis(u_line), Variable(xp.zeros(batchsize, dtype=np.int32)))
        loss = alpha * (loss_fake_p + loss_real_p) + beta * (loss_fake_u + loss_real_u)
        chainer.report({'loss': loss, 'fake_p': loss_fake_p, 'real_p':loss_real_p, 'fake_u': loss_fake_u, 'real_u':loss_real_u}, dis)
        return loss

    def update_core(self):
        xp = self.gen.xp
        self._iter += 1

        batch_p = self.get_iterator('main').next()
        batch_u = self.get_iterator('unpaired').next()
        batchsize = len(batch_u)

        w_in = 256
        w_out = 256

        p_rough = xp.zeros((batchsize, 1, w_in, w_in)).astype("f")
        p_line = xp.zeros((batchsize, 1, w_out, w_out)).astype("f")
        u_rough = xp.zeros((batchsize, 1, w_in, w_in)).astype("f")
        u_line = xp.zeros((batchsize, 1, w_out, w_out)).astype("f")

        for i in range(batchsize):            
            p_rough[i, :] = xp.asarray(batch_p[i][0])
            p_line[i, :] = xp.asarray(batch_p[i][1])

        for i in range(batchsize):
            u_rough[i, :] = xp.asarray(batch_u[i][0])
            u_line[i, :] = xp.asarray(batch_u[i][1])

        p_rough = Variable(p_rough)
        p_line = Variable(p_line)
        u_rough = Variable(u_rough)
        u_line = Variable(u_line)

        G_p_rough = self.gen(p_rough, test=False)
        G_u_rough = self.gen(u_rough, test=False)

        D_p_rough = self.dis(G_p_rough, test=False)
        D_u_rough = self.dis(G_u_rough, test=False)

        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')

        gen_optimizer.update(self.loss_gen, self.gen, G_p_rough, D_p_rough, p_line, D_u_rough, batchsize)
        G_p_rough.unchain_backward()
        dis_optimizer.update(self.loss_dis, self.dis, D_p_rough, p_line, D_u_rough, u_line, batchsize)

if __name__ == '__main__':
    main()

