from util import *
from loss import *
from augument import *
from segmentation_models.segmentation_models import Unet
from segmentation_models.segmentation_models.utils import set_trainable
from keras.backend.tensorflow_backend import set_session
import os
gpu_control('1',0.8)


# 数据导入
train_df = data_initial()

# 构造k折交叉验证
K_flods = 6
x_train,x_valid,y_train,y_valid,cov_train,cov_test,depth_train,depth_test = \
    k_folds_raw(train_df, K_flods, img_size=128,padd=upsample_128, is_single_test=False)
print(np.array(x_train).shape)
#test 数据集划分是否正确
#plot_flods_coverage(train_df,cov_train,flods_num=5,mode='train',is_single_test=True)
#plot_flods_coverage(train_df,cov_test,flods_num=5,mode='test',is_single_test=True)
print('SET split sucessful！')

# 模型训练
history_all = []
num_of_flods =1

for i in range(1):
    save_path = '/home/zhangs/lyc/salt/trained_models/res/flod%d_res34_bce_adam_hp+scse+aug.model' % i
    x_train_flods, y_train_flods = flip(x_train[i], y_train[i])
    x_valid_flods, y_valid_flods = x_valid[i],y_valid[i]
    seq = iaa.Sequential([
        # iaa.Fliplr(0.5),
        iaa.SomeOf((2, 3), [
            iaa.Affine(rotate=(-10, 10), translate_percent={"x": (-0.25, 0.25)}, mode='symmetric', cval=(0)),
            iaa.Affine(shear=(-16, 16), mode='symmetric', cval=(0)),
            iaa.Crop(percent=(0.1, 0.5), keep_size=True),
        ]),
    ])
    x_train_flods_l,y_train_flods_l=[],[]
    for t in range(6):
        seq_det = seq.to_deterministic()
        indexes = random.sample(range(len(x_train_flods)), 1000)
        x_train_flods_aug, y_train_flods_aug = do_augmentation(seq_det, x_train_flods[indexes], y_train_flods[indexes])
        x_train_flods_l+=list(x_train_flods_aug)
        y_train_flods_l+=list(y_train_flods_aug)
    print(np.array(x_train_flods_l).shape)
    print(x_train_flods.shape)

    x_train_flods = np.append(x_train_flods, np.array(x_train_flods_l), axis=0)
    y_train_flods = np.append(y_train_flods, np.array(y_train_flods_l), axis=0)
    print(x_train_flods.shape)
    print('Data augument sucessful！')
    #
    model = Unet(unet_v = 'v1',input_shape=(128,128,3),decoder_filters=(64,32,32,32,16),backbone_name='resnet18',
                 encoder_weights='imagenet',decoder_block_type='upsampling_v3',activation='sigmoid',
                 freeze_encoder=False,decoder_use_batchnorm=True,dp=0.5,sp_dp=0,
                 use_middle=False,use_hypercolum=True,use_scse=False)
    adam = Adam(lr=0.001)
    model.summary()
    model.compile(loss="binary_crossentropy", optimizer=adam, metrics=[my_iou_metric])
    model_checkpoint = ModelCheckpoint(save_path,monitor='val_my_iou_metric',
                                       mode = 'max',  save_best_only=True, verbose=1)
    x_valid_flods = add_depth(x_valid_flods)

    history = model.fit(add_depth(x_train_flods),
                        y_train_flods,
                        validation_data=(x_valid_flods,
                                         y_valid_flods),
                        epochs=50,
                        batch_size=32,
                        verbose=1,
                        callbacks=[model_checkpoint])
    # history = model.fit_generator(generator(x_train_flods, y_train_flods, 32,
    #                                         seq_aug=affine_seq,use_depth=True),
    #                               epochs=50,
    #                               steps_per_epoch=300,
    #                               validation_data=(x_valid_flods, y_valid_flods),
    #                               verbose=1,
    #                               callbacks=[model_checkpoint]
    #                               )

    history_all.append(history)
    custom = {'my_iou_metric': my_iou_metric}
    preds_valid, y_valid_true = predit_with_one_fold_raw(save_path, x_valid_flods,
                                                         y_valid_flods, custom, use_fliptta=True)
    iou_best, threshold_best = thresholds_select(preds_valid, y_valid_true)
    notta_preds, notta_y = predit_with_one_fold_raw(save_path, x_valid_flods,
                                                    y_valid_flods, custom, use_fliptta=False)
    iou = iou_metric_batch(notta_y, np.int32(notta_preds > 0.5))
    f = open('/home/zhangs/lyc/salt/code/logs_res34.txt', "a+")
    f.write( 'flods%d ： best iou=' % i + str(iou_best) + '  threshold best=' + str(threshold_best) +
            '   notta score:' + str(iou) + '\n')
    f.close()
    ############################# finetune ###################################
    save_path2 = '/home/zhangs/lyc/salt/trained_models/res/flod%d_res34_lovasz_adam_hp+scse.model' % i
    model1 = load_model(save_path, custom_objects={'my_iou_metric': my_iou_metric})
    # remove layter activation layer and use losvasz loss
    input_x = model1.layers[0].input
    output_layer = model1.layers[-1].input
    model = Model(input_x, output_layer)
    c = Adam(lr=0.0005)

    # lovasz_loss need input range (-∞，+∞), so cancel the last "sigmoid" activation
    # Then the default threshod for pixel prediction is 0 instead of 0.5, as in my_iou_metric_2.
    model.compile(loss=lovasz_loss, optimizer=c, metrics=[my_iou_metric_2])
    model_checkpoint = ModelCheckpoint(save_path2, monitor='val_my_iou_metric_2',
                                       mode='max', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric_2',mode = 'max',
                                        factor=0.5, patience=10, min_lr=0.00001, verbose=1)
    # history2 = model.fit_generator(generator(x_train_flods, y_train_flods, 32,
    #                                         seq=norml_seq,use_depth=True),
    #                               epochs=100,
    #                               steps_per_epoch=300,
    #                               validation_data=(x_valid_flods, y_valid_flods),
    #                               verbose=1,
    #                               callbacks=[model_checkpoint,reduce_lr]
    #                               )
    history2 = model.fit(add_depth(x_train_flods),
                         y_train_flods,
                         validation_data=(x_valid_flods,
                                          y_valid_flods),
                         epochs=100,
                         batch_size=32,
                         shuffle=True,
                         verbose=1,
                         callbacks=[model_checkpoint])
    custom = {'my_iou_metric_2': my_iou_metric_2, 'lovasz_loss': lovasz_loss}
    preds_valid, y_valid_true = predit_with_one_fold_raw(save_path2, x_valid_flods, y_valid_flods, custom)
    iou_best, threshold_best = thresholds_select(preds_valid, y_valid_true, begin=-0.5, end=0.5)
    notta_preds_lovas, notta_y_lovas = predit_with_one_fold_raw(save_path2, x_valid_flods,
                                                                y_valid_flods, custom, use_fliptta=False)
    iou = iou_metric_batch(notta_y_lovas, np.int32(notta_preds_lovas > 0))
    f = open('/home/zhangs/lyc/salt/code/logs_res34.txt', "a+")
    f.write(data + 'lovasz--' + 'flods%d ： best iou=' % i + str(iou_best) + '  threshold best=' + str(threshold_best) +
            '   notta score:' + str(iou) + '\n')
    f.close()

for i in range(num_of_flods):
    history = history_all[i]
    axs[i][0].plot(history.epoch, history.history["loss"], label="Train loss")
    axs[i][0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    axs[i][1].plot(history.epoch, history.history['my_iou_metric'], label="Train iou")
    axs[i][1].plot(history.epoch, history.history['val_'+'my_iou_metric'], label="Validation iou")
plt.show()
