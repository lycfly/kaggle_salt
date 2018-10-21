import numpy as np
from imgaug import augmenters as iaa
from util import *
from loss import *
import numpy as np
import random
def flip(x_train,y_train):
    x_l_r_flip = [np.fliplr(x) for x in x_train]
    y_l_r_flip = [np.fliplr(x) for x in y_train]

    x_train = np.append(x_train, x_l_r_flip, axis=0)
    y_train = np.append(y_train, y_l_r_flip, axis=0)
    return x_train,y_train
affine_seq = iaa.Sequential([
 # General
 iaa.SomeOf((1, 2),
       [iaa.Fliplr(0.5),
        iaa.Affine(rotate=(-10, 10),
                   translate_percent={"x": (-0.05, 0.05)},
                   mode='edge'),
        # iaa.CropAndPad(percent=((0.0, 0.0), (0.05, 0.0), (0.0, 0.0), (0.05, 0.0)))
        ]),
# Deformations
iaa.Sometimes(0.3, iaa.PiecewiseAffine(scale=(0.04, 0.08))),
iaa.Sometimes(0.3, iaa.PerspectiveTransform(scale=(0.05, 0.1))),
], random_order=True)

intensity_seq = iaa.Sequential([
iaa.Invert(0.3),
iaa.Sometimes(0.3, iaa.ContrastNormalization((0.5, 1.5))),
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
])
], random_order=False)

norml_seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.SomeOf((1, 2), [
                iaa.Noop(),
                iaa.Noop(),
                iaa.Affine(rotate=(-10, 10), translate_percent={"x": (-0.25, 0.25)}, mode='symmetric', cval=(0)),
                iaa.Affine(shear=(-16, 16), mode='symmetric', cval=(0)),
                iaa.Crop(percent=(0.1, 0.5), keep_size=True),
            ]),
        ])
def generator(features, labels, batch_size,seq=affine_seq,use_depth=False,use_repeat=False):
    # create empty arrays to contain batch of features and labels
    batch_features = np.zeros((batch_size, features.shape[1], features.shape[2], features.shape[3]))
    batch_labels = np.zeros((batch_size, labels.shape[1], labels.shape[2], labels.shape[3]))
    while True:
        # seq = seq_aug
        seq_det = seq.to_deterministic()
        # Fill arrays of batch size with augmented data taken randomly from full passed arrays
        indexes = random.sample(range(len(features)), batch_size)
        # Perform the exactly the same augmentation for X and y
        random_augmented_images, random_augmented_labels = do_augmentation(seq_det, features[indexes], labels[indexes])
        batch_features[:,:,:,:] = random_augmented_images[:,:,:,:]
        batch_labels[:,:,:,:] = random_augmented_labels[:,:,:,:]
        if use_depth:
            batch_features = add_depth(batch_features)
        if use_repeat:
            batch_features = repeat(batch_features)
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


