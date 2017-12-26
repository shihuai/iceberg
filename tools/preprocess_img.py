import numpy as np
np.random.seed(666)
import pandas as pd
import os
import cv2
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma, denoise_tv_bregman, denoise_nl_means)

def denoise(X, weight, multichannel):
    return np.asarray([denoise_tv_chambolle(item, weight=weight,
                                            multichannel=multichannel) for item in X])

def get_scaled_imgs(df):
    imgs = []
    r_mean = np.zeros((75, 75))
    g_mean = np.zeros((75, 75))
    b_mean = np.zeros((75, 75))
    for i, row in df.iterrows():
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = ((band_1 + band_2) / 2);
        r_mean = r_mean + band_1
        g_mean = g_mean + band_2
        b_mean = b_mean + band_3

        # a = (band_1 - band_1.min()) / (band_1.max() - band_1.min())
        # b = (band_2 - band_2.min()) / (band_2.max() - band_2.min())
        # c = (band_3 - band_3.min()) / (band_3.max() - band_3.min())

        imgs.append(np.dstack((band_1, band_2, band_3)))

    r_mean = r_mean / len(imgs)
    g_mean = g_mean / len(imgs)
    b_mean = b_mean / len(imgs)
    imgs = np.array(imgs)
    imgs = denoise(imgs, weight=0.05, multichannel=True)
    imgs[:, :, :, 0] = imgs[:, :, :, 0] - r_mean
    imgs[:, :, :, 1] = imgs[:, :, :, 1] - g_mean
    imgs[:, :, :, 2] = imgs[:, :, :, 2] - b_mean
    return np.array(imgs)

def resize_img(src_imgs, height=75, width=75):
    dst_imgs = []
    for img in src_imgs:
        img = cv2.resize(img, (height, width), interpolation=cv2.INTER_LINEAR)
        dst_imgs.append(img)

    return np.array(dst_imgs)

def data_augment(imgs, labels, angles):
    new_imgs = []
    new_labels = []
    new_angles = []
    for idx in range(len(imgs)):
        img = imgs[idx, :, :, :]
        label = labels[idx]
        angle = angles[idx]

        flip_img = cv2.flip(img, 1)
        new_imgs.append(img)
        new_imgs.append(flip_img)
        new_labels.append(label)
        new_labels.append(label)
        new_angles.append(angle)
        new_angles.append(angle)

    return np.array(new_imgs), np.array(new_labels), np.array(new_angles)

def get_channel_mean(imgs, ch_idx):
    mean = 0.0
    pixel_num = imgs.shape[0] * imgs.shape[1] * imgs.shape[2]
    for img in imgs:
        mean = mean + img[:, :, ch_idx].sum() / pixel_num

    return mean

def reduce_mean(imgs):

    if not os.path.exists('./data/means.npy'):
        print 'the means file does not exit.'
        r_mean = get_channel_mean(imgs, 0)
        g_mean = get_channel_mean(imgs, 1)
        b_mean = get_channel_mean(imgs, 2)

        rgb_mean = np.array([r_mean, g_mean, b_mean])
        np.save('means.npy', rgb_mean)
        print rgb_mean
    else:
        rgb_mean = np.load('./data/means.npy')
        print 'have loaded the means file.'
        print rgb_mean

    imgs[:, :, :, 0] = imgs[:, :, :, 0] - rgb_mean[0]
    imgs[:, :, :, 1] = imgs[:, :, :, 1] - rgb_mean[1]
    imgs[:, :, :, 2] = imgs[:, :, :, 2] - rgb_mean[2]

    return imgs

def read_data(file_path=None, height=0, width=0, train_mode=False):
    df = pd.read_json(file_path)
    df.inc_angle = df.inc_angle.replace('na', 0)
    df.inc_angle = df.inc_angle.astype(float).fillna(0.0)

    # imgs = get_scaled_imgs(df)

    x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df['band_1']])
    x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df['band_2']])
    imgs = np.concatenate([x_band1[:, :, :, np.newaxis],
                             x_band2[:, :, :, np.newaxis],
                             ((x_band1 + x_band2) / 2)[:, :, :, np.newaxis]],
                             axis=-1)
    imgs = denoise(imgs, weight=0.05, multichannel=True)
    angle = np.array(df.inc_angle)

    if height != 0 and width != 0:
        imgs = resize_img(imgs, height, width)

    if train_mode:
        y_train = np.array(df['is_iceberg'])
    else:
        y_train = None

    imgs = reduce_mean(imgs)

    return df['id'], imgs, angle, y_train
