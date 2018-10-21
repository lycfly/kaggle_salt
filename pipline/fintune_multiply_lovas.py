from util import *
from loss import *
from augument import *
from segmentation_models.segmentation_models import Unet
from segmentation_models.segmentation_models.utils import set_trainable
from keras.backend.tensorflow_backend import set_session
import os
gpu_control('0',0.3)
# 数据导入
train_df = data_initial()

# 构造k折交叉验证
K_flods = 6
x_train,x_valid,y_train,y_valid,cov_train,cov_test,depth_train,depth_test = \
                            k_folds_raw(train_df,K_flods,padd = upsample_raw,is_single_test=False)

# test 数据集划分是否正确
# plot_flods_coverage(train_df,cov_train,flods_num=6,mode='train',is_single_test=False)
# plot_flods_coverage(train_df,cov_test,flods_num=6,mode='test',is_single_test=False)
print('SET split sucessful！')


# 模型训练
history_all = []
num_of_flods =K_flods

data_raw = '2018.10.9'
data = '2018.10.12'
for i in range(5,6):

    # 数据增强
    x_train_flods, y_train_flods = x_train[i], y_train[i]
    x_valid_flods, y_valid_flods = x_valid[i],y_valid[i]
    print('Data augument sucessful！')

    save_path_raw = '/home/zhangs/lyc/salt/trained_models/'+data_raw+'/flod%d_single_lovasz_adam.model'%i
    save_path_next = '/home/zhangs/lyc/salt/trained_models/'+data+'/flod%d_single_lovasz_adam_morefintune.model'%i
    custom = {'my_iou_metric_2': my_iou_metric_2, 'lovasz_loss': lovasz_loss}
    model = load_model(save_path_raw, custom_objects=custom)
    # remove layter activation layer and use losvasz loss
    c = Adam(lr=0.0005)

    # lovasz_loss need input range (-∞，+∞), so cancel the last "sigmoid" activation
    # Then the default threshod for pixel prediction is 0 instead of 0.5, as in my_iou_metric_2.
    model.compile(loss=lovasz_loss, optimizer=c, metrics=[my_iou_metric_2])
    model_checkpoint = ModelCheckpoint(save_path_next, monitor='val_my_iou_metric_2',
                                       mode='max', save_best_only=True, verbose=1)
    # reduce_lr = ReduceLROnPlateau(monitor='val_'+metric_str,mode = 'max',
    #                                    factor=0.5, patience=6, min_lr=0.00001, verbose=1)
    # history = model.fit(x_train_flods,
    #                     y_train_flods,
    #                     validation_data=(x_valid_flods,
    #                                      y_valid_flods),
    #                     epochs=50,
    #                     batch_size=32,
    #                     verbose=1,
    #                     callbacks=[model_checkpoint])
    history = model.fit_generator(generator(x_train_flods, y_train_flods, 32,seq_aug=affine_seq),
                                  epochs=200,
                                  steps_per_epoch=300,
                                  validation_data=(x_valid_flods, y_valid_flods),
                                  verbose=2,
                                  callbacks=[model_checkpoint]
                                  )
    preds_valid, y_valid_true = predit_with_one_fold_raw(save_path_next, x_valid_flods, y_valid_flods, custom)
    iou_best, threshold_best = thresholds_select(preds_valid, y_valid_true,begin=-0.5,end=0.5)
    f = open('/home/zhangs/lyc/salt/code/logs_moreaug_fintue.txt', "a+")
    f.write(data + 'lovasz-morefinetune-'+'flods%d ： best iou=' % i + str(iou_best) + '  threshold best=' + str(threshold_best) + '\n')
    f.close()
    # model.summary()
    history_all.append(history)
fig, axs = plt.subplots(num_of_flods, 2, squeeze=False, figsize=(15, 5))

for i in range(1):
    history = history_all[i]
    axs[i][0].plot(history.epoch, history.history["loss"], label="Train loss")
    axs[i][0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    axs[i][1].plot(history.epoch, history.history['my_iou_metric'], label="Train iou")
    axs[i][1].plot(history.epoch, history.history['val_'+'my_iou_metric'], label="Validation iou")
plt.show()


