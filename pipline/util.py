
import numpy as np
import pandas as pd
import tensorflow as tf
from random import randint
import cv2
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")
from self_model import *
from sklearn.model_selection import train_test_split

from skimage.transform import resize

from keras.preprocessing.image import load_img
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam,SGD
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout ,BatchNormalization,Activation,Add
from keras import backend as K
from tqdm import tqdm,tnrange
from skimage.util import pad
from sklearn.model_selection import StratifiedKFold
import os
from keras.backend.tensorflow_backend import set_session

# 准备
img_size_ori = 101
img_size_target = 128
def gpu_control(id,percent):
    os.environ["CUDA_VISIBLE_DEVICES"] = id
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = percent
    set_session(tf.Session(config=config))
def upsample_128(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (128, 128), mode='constant', preserve_range=True)
def upsample_raw(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (101, 101), mode='constant', preserve_range=True)
def upsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)
    #res = np.zeros((img_size_target, img_size_target), dtype=img.dtype)
    #res[:img_size_ori, :img_size_ori] = img
    #return res
def upsample_v2(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (256, 256), mode='constant', preserve_range=True)
    #res = np.zeros((img_size_target, img_size_target), dtype=img.dtype)
    #res[:img_size_ori, :img_size_ori] = img
    #return res    
def reflect_pad(img):
    return pad(resize(img, (101*2, 101*2), mode='constant', preserve_range=True),27,'reflect')
def edge_pad(img):
    return pad(resize(img, (101*2, 101*2), mode='constant', preserve_range=True),27,'edge')

def downsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)
    #return img
def get_iou_vector(A, B):
    batch_size = A.shape[0]
    # modify
    A = np.array([downsample(x) for x in A])
    B = np.array([downsample(x) for x in B])
    metric = []
    for batch in range(batch_size):
        t, p = A[batch]>0, B[batch]>0
        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = (np.sum(intersection > 0) + 1e-10 )/ (np.sum(union > 0) + 1e-10)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    return np.mean(metric)

def my_iou_metric(label, pred):
    return tf.py_func(get_iou_vector, [label, pred>0.5], tf.float64)

def my_iou_metric_2(label, pred):
    return tf.py_func(get_iou_vector, [label, pred >0], tf.float64)

def get_mask_type(mask):
    border = 10
    outer = np.zeros((101-2*border, 101-2*border), np.float32)
    outer = cv2.copyMakeBorder(outer, border, border, border, border, borderType = cv2.BORDER_CONSTANT, value = 1)

    cover = (mask>0.5).sum()
    if cover < 8:
        return 0 # empty
    if cover == ((mask*outer) > 0.5).sum():
        return 1 #border
    if np.all(mask==mask[0]):
        return 2 #vertical

    percentage = cover/(101*101)
    if percentage < 0.15:
        return 3
    elif percentage < 0.25:
        return 4
    elif percentage < 0.50:
        return 5
    elif percentage < 0.75:
        return 6
    else:
        return 7

def histcoverage(coverage):
    histall = np.zeros((1,8))
    for c in coverage:
        histall[0,c] += 1
    return histall
def data_initial_coverage_new():
    train_df = pd.read_csv("/home/zhangs/lyc/salt/data/train.csv", index_col="id", usecols=[0])
    depths_df = pd.read_csv("/home/zhangs/lyc/salt/data/depths.csv", index_col="id")
    train_df = train_df.join(depths_df)
    test_df = depths_df[~depths_df.index.isin(train_df.index)]#将生成id不在train中的样本id集合
    train_df["images"] = [np.array(load_img("/home/zhangs/lyc/salt/data/train/images/{}.png".format(idx), grayscale=True))/ 255 for idx in tqdm(train_df.index)]
    train_df["masks"] = [np.array(load_img("/home/zhangs/lyc/salt/data/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm(train_df.index)]
    train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)
    train_df["coverage_class"] = train_df.masks.map(get_mask_type)
    # 将深度信息放入训练图像
    MAX_DEPTH = max(train_df["z"])
    train_df["depth"] = [np.ones_like(train_df.loc[i]["images"]) * train_df.loc[i]["z"] / MAX_DEPTH for i in tqdm(train_df.index)]

    # Image in layer1 + depth in layer2
    train_df["images_d"] = [np.dstack((train_df["images"][i],train_df["depth"][i])) for i in tqdm(train_df.index)]
    # Free up some RAM
    del depths_df
    return train_df

def data_initial():
    train_df = pd.read_csv("/home/zhangs/lyc/salt/data/train.csv", index_col="id", usecols=[0])
    depths_df = pd.read_csv("/home/zhangs/lyc/salt/data/depths.csv", index_col="id")
    train_df = train_df.join(depths_df)
    test_df = depths_df[~depths_df.index.isin(train_df.index)]#将生成id不在train中的样本id集合
    train_df["images"] = [np.array(load_img("/home/zhangs/lyc/salt/data/train/images/{}.png".format(idx), grayscale=True))/ 255 for idx in tqdm(train_df.index)]
    train_df["masks"] = [np.array(load_img("/home/zhangs/lyc/salt/data/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm(train_df.index)]
    train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)
    def cov_to_class(val):
        for i in range(0, 11):
            if val * 10 <= i :
                return i
    train_df["coverage_class"] = train_df.coverage.map(cov_to_class)
    # 将深度信息放入训练图像
    MAX_DEPTH = max(train_df["z"])
    train_df["depth"] = [np.ones_like(train_df.loc[i]["images"]) * train_df.loc[i]["z"] / MAX_DEPTH for i in tqdm(train_df.index)]

    # Image in layer1 + depth in layer2
    train_df["images_d"] = [np.dstack((train_df["images"][i],train_df["depth"][i])) for i in tqdm(train_df.index)]
    # Free up some RAM
    del depths_df
    return train_df
def k_folds_raw(train_df,K_flods,padd = upsample,img_size=101,is_single_test=True):
    X = train_df.index.values
    y = train_df.coverage_class
    skf = StratifiedKFold(n_splits=K_flods,random_state=1337)
    skf.get_n_splits(X, y)
    ids_train,ids_valid,x_train, x_valid, y_train, y_valid, cov_train, \
    cov_test, depth_train, depth_test=[[] for x in range(10)]
    X_whole = np.array(train_df.images.map(padd).tolist()).reshape(-1, img_size, img_size, 1)
    y_whole = np.array(train_df.masks.map(padd).tolist()).reshape(-1, img_size, img_size, 1)
    for i, [train_index, test_index] in enumerate(skf.split(X, y)):
        print("the %dth flod:" % i)
        print("TRAIN:", train_index.shape, "TEST:", test_index.shape)
        ids_train.append(X[train_index])
        ids_valid.append(X[test_index])
        #
        x_train.append(X_whole[train_index])
        x_valid.append(X_whole[test_index])
        #
        y_train.append(y_whole[train_index])
        y_valid.append(y_whole[test_index])
        #
        cov_train.append(train_df.coverage.values[train_index])
        cov_test.append(train_df.coverage.values[test_index])
        #
        depth_train.append(train_df.z.values[train_index])
        depth_test.append(train_df.z.values[test_index])

        if is_single_test:
            x_train, x_valid, y_train, y_valid ,cov_train,cov_test,depth_train,depth_test= \
                x_train[0],x_valid[0],y_train[0],y_valid[0],cov_train[0],cov_test[0],depth_train[0],depth_test[0]
            break

    del X_whole, y_whole
    return x_train,x_valid,y_train,y_valid,cov_train,cov_test,depth_train,depth_test

def k_folds(train_df,K_flods,padd = reflect_pad,is_single_test=True):
    X = train_df.index.values
    y = train_df.coverage_class
    skf = StratifiedKFold(n_splits=K_flods,random_state=1337)
    skf.get_n_splits(X, y)
    ids_train,ids_valid,x_train, x_valid, y_train, y_valid, cov_train, \
    cov_test, depth_train, depth_test=[[] for x in range(10)]
    X_whole = np.array(train_df.images.map(padd).tolist()).reshape(-1, 256, 256, 1)
    y_whole = np.array(train_df.masks.map(padd).tolist()).reshape(-1, 256, 256, 1)
    for i, [train_index, test_index] in enumerate(skf.split(X, y)):
        print("the %dth flod:" % i)
        print("TRAIN:", train_index.shape, "TEST:", test_index.shape)
        ids_train.append(X[train_index])
        ids_valid.append(X[test_index])
        #
        x_train.append(X_whole[train_index])
        x_valid.append(X_whole[test_index])
        #
        y_train.append(y_whole[train_index])
        y_valid.append(y_whole[test_index])
        #
        cov_train.append(train_df.coverage.values[train_index])
        cov_test.append(train_df.coverage.values[test_index])
        #
        depth_train.append(train_df.z.values[train_index])
        depth_test.append(train_df.z.values[test_index])

        if is_single_test:
            x_train, x_valid, y_train, y_valid ,cov_train,cov_test,depth_train,depth_test= \
                x_train[0],x_valid[0],y_train[0],y_valid[0],cov_train[0],cov_test[0],depth_train[0],depth_test[0]
            break

    del X_whole, y_whole
    return x_train,x_valid,y_train,y_valid,cov_train,cov_test,depth_train,depth_test
#    print(len(x_train))
#   print(x_train[0].shape)

# 画各个flods的salt分布图，检验k-flods是否正确
# usage：
#plot_flods_coverage(cov_train,flods_num=5,mode='train')
#plot_flods_coverage(cov_test,flods_num=5,mode='test')
def plot_flods_coverage(train_df,cov,flods_num=5,mode='train',is_single_test=True):
    if is_single_test:
        fig, axs = plt.subplots(1, 1+1, figsize=(15,5))
        sns.distplot(train_df.coverage, kde=False, ax=axs[0])
        for i in range(1,1+1):
            sns.distplot(cov, bins=10, kde=False, ax=axs[i])
            axs[i].set_xlabel("Coverage of k%d"%(0))
        plt.suptitle("Salt coverage of single_flod "+mode)
        axs[0].set_xlabel("Coverage")
        plt.show()
    else:
        fig, axs = plt.subplots(1, flods_num+1, figsize=(15,5))
        sns.distplot(train_df.coverage, kde=False, ax=axs[0])
        for i in range(1,flods_num+1):
            sns.distplot(cov[i-1], bins=10, kde=False, ax=axs[i])
            axs[i].set_xlabel("Coverage of k%d"%(i-1))
        plt.suptitle("Salt coverage of k-flods "+mode)
        axs[0].set_xlabel("Coverage")
        plt.show()

def predit_with_kfolds(K_flods,x_img):
    preds_valid_all = []
    for i in range(K_flods):
        model_flods = load_model("trained_models/%dth_flod.model"%i, custom_objects={'mean_iou': mean_iou})
    #     model.append(model_flods)
        #此处的validation为第0组flod
        preds_valid_flods = model_flods.predict(np.repeat(x_img,3,axis=-1))
        print(preds_valid_flods.shape)
        preds_valid_flods = np.array([downsample(x) for x in preds_valid_flods[:,27:229,27:229,:]])
        print(preds_valid_flods.shape)
        preds_valid_all.append(preds_valid_flods)
    preds_valid = (preds_valid_all[0]+preds_valid_all[1]+preds_valid_all[2]+preds_valid_all[3]+preds_valid_all[4])/5
    return preds_valid

def predit_with_one_fold_raw(model,x_img,y_img,custom,is_test=False,use_fliptta=True):
    if is_test:
        model_flods = model
    else:
        model_flods = load_model(model, custom_objects=custom)
    preds_valid = model_flods.predict(x_img)
    preds_valid = np.array([downsample(x) for x in preds_valid])
    if use_fliptta:
        x_img_reflect = np.array([np.fliplr(x) for x in x_img])
        preds_valid_reflect = model_flods.predict(x_img_reflect)
        preds_valid_reflect = np.array([ np.fliplr(downsample(x)) for x in preds_valid_reflect])
        preds_avg = (preds_valid +preds_valid_reflect)/2
    else:
        preds_avg = preds_valid
    if is_test:
        return preds_avg
    else:
        y_valid = np.array([downsample(x) for x in y_img])
        return preds_avg,y_valid

def predit_with_one_fold(model,x_img,y_img,custom):
    model_flods = load_model(model, custom_objects=custom)
    x_img_reflect = np.array([np.fliplr(x) for x in x_img])
    preds_valid = model_flods.predict(np.repeat(x_img,3,axis=-1))
    preds_valid = np.array([downsample(x) for x in preds_valid[:,27:229,27:229,:]])

    preds_valid_reflect = model_flods.predict(np.repeat(x_img_reflect,3,axis=-1))
    preds_valid_reflect = np.array([downsample(x) for x in preds_valid_reflect[:,27:229,27:229,:]])
    preds_valid_reflect = np.array([ np.fliplr(x) for x in preds_valid_reflect] )
    preds_avg = (preds_valid +preds_valid_reflect)/2

    y_valid = np.array([downsample(x) for x in y_img[:,27:229,27:229,:]])
    return preds_avg,y_valid

def predit_with_one_fold_test(model,x_img):
    x_img_reflect = np.array([np.fliplr(x) for x in x_img])
    preds_valid = model.predict(np.repeat(x_img, 3, axis=-1))
    preds_valid = np.array([downsample(x) for x in preds_valid[:, 27:229, 27:229, :]])

    preds_valid_reflect = model.predict(np.repeat(x_img_reflect, 3, axis=-1))
    preds_valid_reflect = np.array([downsample(x) for x in preds_valid_reflect[:, 27:229, 27:229, :]])
    preds_valid_reflect = np.array([np.fliplr(x) for x in preds_valid_reflect])
    preds_avg = (preds_valid + preds_valid_reflect) / 2

    return preds_avg

def predit_with_one_fold_test_with_depth(model,x_img):
    x_img_reflect = np.array([np.fliplr(x) for x in x_img])
    preds_valid = model.predict(add_depth(x_img))
    preds_valid = np.array([downsample(x) for x in preds_valid[:, 27:229, 27:229, :]])

    preds_valid_reflect = model.predict(add_depth(x_img_reflect))
    preds_valid_reflect = np.array([downsample(x) for x in preds_valid_reflect[:, 27:229, 27:229, :]])
    preds_valid_reflect = np.array([np.fliplr(x) for x in preds_valid_reflect])
    preds_avg = (preds_valid + preds_valid_reflect) / 2

    return preds_avg

def predit_with_one_fold_with_depth(model,x_img,y_img,custom):
    model_flods = load_model(model, custom_objects=custom)
    x_img_reflect = np.array([np.fliplr(x) for x in x_img])
    preds_valid = model_flods.predict(add_depth(x_img))
    preds_valid = np.array([downsample(x) for x in preds_valid[:,27:229,27:229,:]])

    preds_valid_reflect = model_flods.predict(add_depth(x_img_reflect))
    preds_valid_reflect = np.array([downsample(x) for x in preds_valid_reflect[:,27:229,27:229,:]])
    preds_valid_reflect = np.array([ np.fliplr(x) for x in preds_valid_reflect] )
    preds_avg = (preds_valid +preds_valid_reflect)/2

    y_valid = np.array([downsample(x) for x in y_img[:,27:229,27:229,:]])
    return preds_avg,y_valid
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in

    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1  # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)

    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)


def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)

def thresholds_select(preds_valid,y_valid,begin=0,end=1):
    thresholds = np.linspace(begin, end, 50)
    ious = np.array([iou_metric_batch(y_valid, np.int32(preds_valid > threshold)) for threshold in tqdm(thresholds)])
    threshold_best_index = np.argmax(ious[9:-10]) + 9
    iou_best = ious[threshold_best_index]
    threshold_best = thresholds[threshold_best_index]
    plt.plot(thresholds, ious)
    plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
    plt.xlabel("Threshold")
    plt.ylabel("IoU")
    plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
    plt.legend()
    plt.show()
    return iou_best,threshold_best
def thresholds_select_lovas(preds_valid,y_valid):
    thresholds = np.linspace(-0.5, 0.5, 50)
    ious = np.array([iou_metric_batch(y_valid, np.int32(preds_valid > threshold)) for threshold in tqdm(thresholds)])
    threshold_best_index = np.argmax(ious[9:-10]) + 9
    iou_best = ious[threshold_best_index]
    threshold_best = thresholds[threshold_best_index]
    plt.plot(thresholds, ious)
    plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
    plt.xlabel("Threshold")
    plt.ylabel("IoU")
    plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
    plt.legend()
    plt.show()
    return iou_best,threshold_best
# def _add_depth_channels(image):
#     n, h, w,_= image.shape
#     image = list(image)
#     for i in range(n):
#         for row, const in enumerate(np.linspace(0, 1, h-54)):
#             image[i][row+27, :,1] = const
#         for row in range(27):
#             image[i][row,:,1] = image[i][53-row,:,1]
#             image[i][255-row,:,1] = image[i][202+row,:,1]
#
#         image[i][:,:,2] = image[i][:,:,0] * image[i][:,:,1]
#         # print(image[i][1])
#         # print(image[i][2])
#     image = np.array(image)
#     return image
def _add_depth_channels(image):
    n, h, w,_= image.shape
    image = list(image)
    for i in range(n):
        for row, const in enumerate(np.linspace(0, 1, h)):
            image[i][row, :,1] = const
        image[i][:,:,2] = image[i][:,:,0] * image[i][:,:,1]
        # print(image[i][1])
        # print(image[i][2])
    image = np.array(image)
    return image
def add_depth(img):
    img = np.repeat(img[..., :1], 3, axis=-1)
    img = _add_depth_channels(img)
    return img
def repeat(img):
    img = np.repeat(img[..., :1], 3, axis=-1)
    return img



