# -*- coding: utf-8 -*-
import cv2
import numpy as np
import random
import argparse
import os
import sys
import subprocess

np.set_printoptions(threshold=np.inf)

def aff(img, local=True):
    rows, cols = img.shape
    img_out = img / 255.0
    if local:
        a = random.randint(1, 3)
        for i in range(a):
            img_cp = 1.0 - img / 255.0
            # affine transformation
            if random.random() > 0.7:
                rad = np.pi * random.uniform(-0.03, 0.03)
            else:
                rad = 0
            if random.random() > 0.5:
                move_x = random.uniform(-2,2)
            else:
                move_x = 0
            if random.random() > 0.5:
                move_y = random.uniform(-2,2)
            else:
                move_y = 0
            afn_M = np.float32([[np.cos(rad), -1 * np.sin(rad), move_x],
                                [np.sin(rad), np.cos(rad), move_y]])
            img_cp = cv2.warpAffine(img_cp, afn_M, (cols, rows), flags=cv2.INTER_LINEAR)
            img_out = np.where(img_out < 0.1, img_out, img_out - img_cp / random.uniform(1,2))
    else:
        for i in range(random.randint(1,3)):
            if random.random() < 0.6:
                img_cp = 1.0 - img / 255.0
                # affine transformation
                if random.random() > 0.5:
                    rad = np.pi * random.uniform(-0.08, 0.08)
                else:
                    rad = 0
                if random.random() > 0.4:
                    move_x = random.uniform(-5,5)
                else:
                    move_x = 0
                if random.random() > 0.4:
                    move_y = random.uniform(-5,5)
                else:
                    move_y = 0
                afn_M = np.float32([[np.cos(rad), -1 * np.sin(rad), move_x],
                                    [np.sin(rad), np.cos(rad), move_y]])
                img_cp = cv2.warpAffine(img_cp, afn_M, (cols, rows), flags=cv2.INTER_LINEAR)
                img_out = np.where(img_out < 0.3, img_out, img_out - img_cp / random.uniform(1, 3))
    img_out = np.where(img_out < 0, -img_out, img_out)
    return (img_out * 255).astype(np.uint8)

def addPointNoise(img):
    rows, cols = img.shape
    row = random.randint(0, rows - 1)
    col = random.randint(0, cols - 1)
    for j in range(random.randint(4, 35)):
        #print(row, col)
        img[row, col] = 0
        patern = random.randint(1, 8)
        if patern == 1:
            row = min(row + 1, rows - 1)
        elif patern == 2:
            col = min(col + 1, cols - 1)
        elif patern == 3:
            col = min(col + 1, cols - 1)
            row = min(row + 1, rows - 1)
        elif patern == 4:
            row = max(row - 1, 0)
        elif patern == 5:
            col = max(col - 1, 0)
        elif patern == 6:
            row = max(row - 1, 0)
            col = max(col - 1, 0)
        elif patern == 7:
            row = min(row + 1, rows - 1)
            col = max(col - 1, 0)
        elif patern == 8:
            row = max(row - 1, 0)
            col = min(col + 1, cols - 1)

def addGaussianNoise(img):
    rows, cols = img.shape
    mean = 0
    sigma = random.randint(2,8)
    gauss = np.random.normal(mean, sigma, (rows, cols))
    gauss = gauss.reshape(rows, cols)
    gauss = np.where(gauss > 0, gauss, -gauss).astype(np.uint8)
    noisy = np.where(img < 30, img + gauss, img - gauss)
    return noisy

def dilation(img, glob=False):
    n = random.randint(1,2)
    if (random.random() < 0.05) and not glob:
        n = 3
    neibor = np.ones((n,n), np.uint8)
    img = cv2.dilate(img, neibor, iterations = 1)
    return img

def erosion(img):
    n = random.randint(1,2)
    neibor = np.ones((n,n), np.uint8)
    img = cv2.erode(img, neibor, iterations = 1)
    return img

def addDirt(img):
    rows, cols = img.shape
    Dirt = np.ones(img.shape, np.uint8) * 255
    change = 0
    for i in range(random.randint(0,3)):
        change = 1
        y = random.randint(0,rows-1)
        x = random.randint(0,cols-1)
        dy = random.randint(rows//30,rows//5)
        dx = random.randint(cols//30,cols//5)
        Dirt[y:min(y+dy,rows-1), x:min(x+dx,cols-1)] =  random.randint(215,230)
    k_size = random.randint(max(rows,cols)//18,max(rows,cols)//14) * 2 + 1
    Dirt = cv2.GaussianBlur(Dirt, (k_size, k_size), 0)
    if change:
        img = np.where(img < 230, img, Dirt)
    return img


def make_rough(src):
    img = cv2.imread(src, 0)
    rows, cols = img.shape
    zero = np.zeros(img.shape, np.uint8)

    # apply affine transformation
    itv = random.randint(min(rows,cols)//30,min(rows,cols)//20) #interval
    i_n = rows//itv
    j_n = cols//itv
    for i in range(i_n-1):
        for j in range(j_n-1):
            zero[i*itv:(i+1)*itv, j*itv:(j+1)*itv] = aff(img[i*itv:(i+1)*itv, j*itv:(j+1)*itv])
        zero[i*itv:(i+1)*itv, (j_n-1)*itv:cols] = aff(img[i*itv:(i+1)*itv, (j_n-1)*itv:cols])
    for j in range(j_n):
        zero[(i_n-1)*itv:rows, j*itv:(j+1)*itv] = aff(img[(i_n-1)*itv:rows, j*itv:(j+1)*itv])
    zero[(i_n-1)*itv:rows, (j_n-1)*itv:cols] = aff(img[(i_n-1)*itv:rows, (j_n-1)*itv:cols])

    # apply larger affine transformation
    itv = random.randint(min(rows,cols)//20,min(rows,cols)//15) #interval
    i_n = rows//itv
    j_n = cols//itv
    for i in range(i_n-1):
        for j in range(j_n-1):
            zero[i*itv:(i+1)*itv, j*itv:(j+1)*itv] = aff(zero[i*itv:(i+1)*itv, j*itv:(j+1)*itv], local=False)
        zero[i*itv:(i+1)*itv, (j_n-1)*itv:cols] = aff(zero[i*itv:(i+1)*itv, (j_n-1)*itv:cols], local=False)
    for j in range(j_n):
        zero[(i_n-1)*itv:rows, j*itv:(j+1)*itv] = aff(zero[(i_n-1)*itv:rows, j*itv:(j+1)*itv], local=False)
    zero[(i_n-1)*itv:rows, (j_n-1)*itv:cols] = aff(zero[(i_n-1)*itv:rows, (j_n-1)*itv:cols], local=False)

    # dilation & erosion
    itv = random.randint(55,80) #interval
    i_n = rows//itv
    j_n = cols//itv
    for i in range(i_n-1):
        for j in range(j_n-1):
            if random.random() < 0.1:
                zero[i*itv:(i+1)*itv, j*itv:(j+1)*itv] = dilation(zero[i*itv:(i+1)*itv, j*itv:(j+1)*itv])
            elif random.random() > 0.8:
                zero[i*itv:(i+1)*itv, j*itv:(j+1)*itv] = erosion(zero[i*itv:(i+1)*itv, j*itv:(j+1)*itv])
        if random.random() < 0.1:
            zero[i*itv:(i+1)*itv, (j_n-1)*itv:cols] = dilation(zero[i*itv:(i+1)*itv, (j_n-1)*itv:cols])
        elif random.random() > 0.8:
            zero[i*itv:(i+1)*itv, (j_n-1)*itv:cols] = erosion(zero[i*itv:(i+1)*itv, (j_n-1)*itv:cols])
    for j in range(j_n):
        if random.random() < 0.1:
            zero[(i_n-1)*itv:rows, j*itv:(j+1)*itv] = dilation(zero[(i_n-1)*itv:rows, j*itv:(j+1)*itv])
        elif random.random() > 0.8:
            zero[(i_n-1)*itv:rows, j*itv:(j+1)*itv] = erosion(zero[(i_n-1)*itv:rows, j*itv:(j+1)*itv])
    if random.random() < 0.1:
        zero[(i_n-1)*itv:rows, (j_n-1)*itv:cols] = dilation(zero[(i_n-1)*itv:rows, (j_n-1)*itv:cols])
    elif random.random() > 0.8:
        zero[(i_n-1)*itv:rows, (j_n-1)*itv:cols] = erosion(zero[(i_n-1)*itv:rows, (j_n-1)*itv:cols])


    if random.random() < 0.15:
        # dilation
        zero = dilation(zero, glob=True)
    elif random.random() > 0.95:
        # erosion
        zero = erosion(zero)

    # add point noise
    for i in range(random.randint(25,40)):
        addPointNoise(zero)

    # blur
    k_size = random.randint(0, 1) * 2 + 1
    blur = cv2.GaussianBlur(zero, (k_size, k_size), 0)
    img = blur

    # add dirt
    img = addDirt(img)

    # reduce contrast
    min_table = random.randint(0,150)
    max_table = 255
    diff_table = max_table - min_table
    look_up_table = np.arange(256, dtype='uint8')
    for i in range(0, 255):
        look_up_table[i] = min_table + i * (diff_table) / 255
    img = cv2.LUT(img.astype(np.uint8), look_up_table)

    # add Gaussian Noise
    img = addGaussianNoise(img)

    return img



parser = argparse.ArgumentParser(description='change line art to rough sketch')
parser.add_argument('directory', help='Path to line art files')
args = parser.parse_args()

def cmd(cmd):
    return subprocess.getoutput(cmd)

if __name__ == '__main__':
    #os.mkdir('rough')
    #os.mkdir('line')
    n_i = 901
    #n_f = 1
    dirs = cmd('ls ' + args.directory)
    images = dirs.splitlines()
    for i in images:
        print('-img: ' + str(n_i) + '/' + str(len(images)))
        cmd('cp ' + args.directory + '/' + i + ' line/{0:07d}.jpg'.format(n_i))
        rough = make_rough(args.directory + '/' + i)
        cv2.imwrite('rough/{0:07d}.jpg'.format(n_i), rough)
        n_i += 1
    #folders = dirs.splitlines()
    #for f in folders:
    #    print('folder:' + str(n_f) + '/' + str(len(folders)))
    #    folds = cmd('ls ' + args.directory + '/' + f)
    #    images = folds.splitlines()
    #    for i in images:
    #        print('-img: ' + str(n_i) + '/' + str(len(images)))
    #        cmd('cp ' + args.directory + '/' + f + '/' + i + ' line/{0:07d}.jpg'.format(n_i))
    #        rough = make_rough(args.directory + '/' + f + '/' + i)
    #        cv2.imwrite('rough/{0:07d}.jpg'.format(n_i), rough)
    #        n_i += 1
    #    n_f += 1