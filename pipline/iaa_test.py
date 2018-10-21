from imgaug import augmenters as iaa
from util import *
from loss import *
import numpy as np
import random
import matplotlib.pyplot as plt
import imgaug as ia
from keras.backend.tensorflow_backend import set_session
import os
# 数据导入
train_df = data_initial()
# 构造k折交叉验证
K_flods = 6
x_train,x_valid,y_train,y_valid,cov_train,cov_test,depth_train,depth_test = \
    k_folds_raw(train_df,K_flods,padd=upsample_raw,is_single_test=True)
print(np.array(x_train).shape)
# Return augmented images/masks arrays of batch size
def generator(features, labels, batch_size, seq):
    # create empty arrays to contain batch of features and labels
    batch_features = np.zeros((batch_size, features.shape[1], features.shape[2], features.shape[3]))
    batch_labels = np.zeros((batch_size, labels.shape[1], labels.shape[2], labels.shape[3]))
    while True:
        seq_det = seq.to_deterministic()
        # Fill arrays of batch size with augmented data taken randomly from full passed arrays
        indexes = random.sample(range(len(features)), batch_size)
        # Perform the exactly the same augmentation for X and y
        random_augmented_images, random_augmented_labels = do_augmentation(seq_det, features[indexes], labels[indexes])
        batch_features[:,:,:,:] = random_augmented_images[:,:,:,:]
        batch_labels[:,:,:,:] = random_augmented_labels[:,:,:,:]

        yield batch_features, batch_labels
def do_augmentation(seq_det, X_train, y_train):
    # Move from 0-1 float to uint8 format (needed for most imgaug operators)
    X_train_aug = [(x[:,:,:] * 255.0).astype(np.uint8) for x in X_train]
    # print(X_train_aug[0].shape)
    # Do augmentation
    X_train_aug = seq_det.augment_images(X_train_aug)
    # Back to 0-1 float range
    X_train_aug = [(x[:,:,:].astype(np.float64)) / 255.0 for x in X_train_aug]

    # Move from 0-1 float to uint8 format (needed for imgaug)
    y_train_aug = [(x[:,:,:] * 255.0).astype(np.uint8) for x in y_train]
    # Do augmentation
    y_train_aug = seq_det.augment_images(y_train_aug)
    # Make sure we only have 2 values for mask augmented
    y_train_aug = [np.where(x[:,:,:] > 0, 255, 0) for x in y_train_aug]
    # Back to 0-1 float range
    y_train_aug = [(x[:,:,:].astype(np.float64)) / 255.0 for x in y_train_aug]
    return np.array(X_train_aug), np.array(y_train_aug)

seq = iaa.Sequential([
     iaa.Fliplr(0.5),
     iaa.SomeOf((1, 2),[
         iaa.Noop(),
         iaa.Noop(),
         iaa.Affine(rotate=(-10, 10),translate_percent={"x": (-0.25, 0.25)}, mode='symmetric', cval=(0)),
         iaa.Affine(shear=(-16, 16), mode='symmetric', cval=(0)),
         iaa.Crop(percent=(0.1, 0.5), keep_size=True),
                ]),
     iaa.OneOf([
        iaa.Invert(0.3),
        iaa.Sometimes(0.3, iaa.ContrastNormalization((0.5, 1.5))),
        iaa.Noop(),
        iaa.Noop(),
        iaa.OneOf([
             iaa.Noop(),
             iaa.Sequential([
                 iaa.OneOf([
                     iaa.Add((-10, 10)),
                     iaa.AddElementwise((-10, 10)),
                     iaa.Multiply((0.95, 1.05)),
                     iaa.MultiplyElementwise((0.95, 1.05)),
                 ]),
             ]),
             iaa.OneOf([
                 iaa.GaussianBlur(sigma=(0.0, 1.0)),
                 iaa.AverageBlur(k=(2, 5)),
                 iaa.MedianBlur(k=(3, 5))
             ])
         ]),
        iaa.Noop(),
        iaa.PerspectiveTransform(scale=(0.04, 0.08)),
        iaa.Noop(),
        iaa.PiecewiseAffine(scale=(0.05, 0.1), mode='edge', cval=(0)),
        ]),
    # More as you want ...
])
# images_aug = []
# fig, axs = plt.subplots(4, 10,squeeze=False, figsize=(15, 5))
# # seq_det = seq.to_deterministic()
# # x_aug,y_aug = do_augmentation(seq_det,x_train[:10],y_train[:10])
# # for i in range(10):
# #     x_aug[i] = upsample_raw(x_aug[i])
# #     y_aug[i] = upsample_raw(y_aug[i])
# x_aug,y_aug = generator(x_train,y_train, 10, seq)
# for i in range(10):
#     axs[0][i].imshow(np.reshape(x_aug[i],(101,101)))
#     axs[1][i].imshow(np.reshape(x_train[i],(101,101)))
#     axs[2][i].imshow(np.reshape(y_aug[i],(101,101)))
#     axs[3][i].imshow(np.reshape(y_train[i],(101,101)))
# plt.show()
