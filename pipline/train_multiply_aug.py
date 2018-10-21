from util import *
from loss import *
from augument import *
from segmentation_models.segmentation_models import Unet
from segmentation_models.segmentation_models.utils import set_trainable
from keras.backend.tensorflow_backend import set_session
import os
gpu_control('1',0.3)

# 数据导入
train_df = data_initial_coverage_new()

# 构造k折交叉验证
K_flods = 5
x_train,x_valid,y_train,y_valid,cov_train,cov_test,depth_train,depth_test = \
                            k_folds_raw(train_df,K_flods,padd = upsample_raw,is_single_test=False)

# test 数据集划分是否正确
# plot_flods_coverage(train_df,cov_train,flods_num=6,mode='train',is_single_test=False)
# plot_flods_coverage(train_df,cov_test,flods_num=6,mode='test',is_single_test=False)
print('SET split sucessful！')


# 模型训练
history_all = []
num_of_flods =K_flods
image_size = 101

data = '2018.10.10'
for i in range(1,3):
    metric = my_iou_metric
    metric_str = 'my_iou_metric'

    custom = {metric_str: metric}
    save_path = '/home/zhangs/lyc/salt/trained_models/'+data+'/flod%d_single_bce_adam.model'%i
    # 数据
    x_train_flods, y_train_flods = x_train[i], y_train[i]
    x_valid_flods, y_valid_flods = x_valid[i],y_valid[i]
    print('Data augument sucessful！')

    input_layer = Input((image_size, image_size, 1))
    output_layer = build_model(input_layer, 16, 0.5)
    model = Model(input_layer, output_layer)
    # model.summary()
    adam = Adam(lr=0.001)
    #sgd  = SGD(lr=0.005, momentum=0.9, decay=0.0001, nesterov=False)
    model.compile(loss="binary_crossentropy", optimizer=adam, metrics=[metric])
    #model.compile(loss=lovasz_loss, optimizer=sgd, metrics=[metric])
    model_checkpoint = ModelCheckpoint(save_path,monitor='val_'+metric_str,
                                       mode = 'max',  save_best_only=True, verbose=1)
    # reduce_lr = ReduceLROnPlateau(monitor='val_'+metric_str,mode = 'max',
    #                                    factor=0.5, patience=6, min_lr=0.00001, verbose=1)
    history = model.fit_generator(generator(x_train_flods, y_train_flods, 32),
                                  epochs=500,
                                  steps_per_epoch=220,
                                  validation_data=(x_valid_flods, y_valid_flods),
                                  verbose=2,
                                  callbacks=[model_checkpoint]
                                  )

    history_all.append(history)

    preds_valid, y_valid_true = predit_with_one_fold_raw(save_path, x_valid_flods, y_valid_flods, custom)
    iou_best, threshold_best = thresholds_select(preds_valid, y_valid_true)
    notta_preds, notta_y = predit_with_one_fold_raw(save_path, x_valid_flods,
                                                    y_valid_flods, custom, use_fliptta=False)
    iou = iou_metric_batch(notta_y, np.int32(notta_preds > 0.5))
    f = open('/home/zhangs/lyc/salt/code/logs2.txt',"a+")
    f.write(data+'flods%d ： best iou='%i+str(iou_best)+'  threshold best='+str(threshold_best)+
            '   notta score:'+str(iou)+'\n')
    f.close()

    save_path2 = '/home/zhangs/lyc/salt/trained_models/'+data+'/flod%d_single_lovasz_adam.model'%i
    model1 = load_model(save_path, custom_objects={'my_iou_metric': my_iou_metric})
    # remove layter activation layer and use losvasz loss
    input_x = model1.layers[0].input

    output_layer = model1.layers[-1].input #
    model = Model(input_x, output_layer)

    c = Adam(lr=0.0005)

    # lovasz_loss need input range (-∞，+∞), so cancel the last "sigmoid" activation
    # Then the default threshod for pixel prediction is 0 instead of 0.5, as in my_iou_metric_2.
    model.compile(loss=lovasz_loss, optimizer=c, metrics=[my_iou_metric_2])
    model_checkpoint = ModelCheckpoint(save_path2, monitor='val_my_iou_metric_2',
                                       mode='max', save_best_only=True, verbose=1)
    # reduce_lr = ReduceLROnPlateau(monitor='val_'+metric_str,mode = 'max',
    #                                    factor=0.5, patience=6, min_lr=0.00001, verbose=1)
    history2 = model.fit_generator(generator(x_train_flods, y_train_flods, 32),
                                  epochs=200,
                                  steps_per_epoch=220,
                                  validation_data=(x_valid_flods, y_valid_flods),
                                  verbose=2,
                                  callbacks=[model_checkpoint]
                                  )

    custom = {'my_iou_metric_2': my_iou_metric_2, 'lovasz_loss': lovasz_loss}
    preds_valid, y_valid_true = predit_with_one_fold_raw(save_path2, x_valid_flods, y_valid_flods, custom)
    iou_best, threshold_best = thresholds_select(preds_valid, y_valid_true,begin=-0.5,end=0.5)
    notta_preds_lovas, notta_y_lovas = predit_with_one_fold_raw(save_path2, x_valid_flods,
                                                                y_valid_flods, custom, use_fliptta=False)
    iou = iou_metric_batch(notta_y_lovas, np.int32(notta_preds_lovas > 0))
    f = open('/home/zhangs/lyc/salt/code/logs2.txt', "a+")
    f.write(data + 'lovasz--'+'flods%d ： best iou=' % i + str(iou_best) + '  threshold best=' + str(threshold_best) +
            '   notta score:' + str(iou) +'\n')
    f.close()
    # model.summary()
fig, axs = plt.subplots(num_of_flods, 2, squeeze=False, figsize=(15, 5))

for i in range(num_of_flods):
    history = history_all[i]
    axs[i][0].plot(history.epoch, history.history["loss"], label="Train loss")
    axs[i][0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    axs[i][1].plot(history.epoch, history.history['my_iou_metric'], label="Train iou")
    axs[i][1].plot(history.epoch, history.history['val_'+'my_iou_metric'], label="Validation iou")
plt.show()


