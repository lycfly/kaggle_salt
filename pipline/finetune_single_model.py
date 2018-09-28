from util import *
from loss import *
from augument import *
from segmentation_models.segmentation_models import Unet
from segmentation_models.segmentation_models.utils import set_trainable
from keras.backend.tensorflow_backend import set_session
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# set_session(tf.Session(config=config))

# 数据导入
train_df = data_initial()

# 构造k折交叉验证
K_flods = 6
x_train,x_valid,y_train,y_valid,cov_train,cov_test,depth_train,depth_test =\
                             k_folds(train_df,K_flods,is_single_test=True)
print(np.array(x_train).shape)
#test 数据集划分是否正确
#plot_flods_coverage(train_df,cov_train,flods_num=5,mode='train',is_single_test=True)
#plot_flods_coverage(train_df,cov_test,flods_num=5,mode='test',is_single_test=True)
print('SET split sucessful！')

# 数据增强
x_train, y_train = flip(x_train,y_train)
print('Data augument sucessful！')
# 模型训练
father_model = '../trained_models/single_res34_dp02.model'
son_model = '../trained_models/single_res34_dp02_son0.model'
history_all = []
num_of_flods =1
for i in range(num_of_flods):

    fig, axs = plt.subplots(num_of_flods, 2, squeeze=False,figsize=(15, 5))
    # 重载父模型
    custom = {'my_iou_metric': my_iou_metric, 'bce_dice_loss': bce_dice_loss}
    model1 = load_model(father_model, custom_objects=custom)
    # lovas
    input_x = model1.layers[0].input
    output_layer = model1.layers[-1].input
    model = Model(input_x, output_layer)
    model.summary()
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    #sgd  = SGD(lr=0.0005, momentum=0.9, decay=0.0001, nesterov=False)
    model.compile(loss=lovasz_loss, optimizer=adam, metrics=[my_iou_metric_2])
    #model.compile(loss=keras_lovasz_hinge, optimizer=sgd, metrics=[my_iou_metric_2])
    early_stopping = EarlyStopping(monitor='val_my_iou_metric_2', mode = 'max', patience=15, verbose=1)
    model_checkpoint = ModelCheckpoint(son_model,monitor='val_my_iou_metric_2',
                                       mode = 'max',  save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric_2',mode = 'max',
                                       factor=0.5, patience=5, min_lr=0.000001, verbose=1)

    history = model.fit(np.repeat(x_train[..., :1],3,axis=-1),
                          y_train,
                          validation_data=(np.repeat(x_valid[..., :1],3,axis=-1),
                          y_valid),
                          epochs=80,
                          batch_size=32,
                          verbose=1,
                          #shuffle=True,
                          callbacks=[model_checkpoint, reduce_lr,early_stopping])
    history_all.append(history)
    axs[i][0].plot(history.epoch, history.history["loss"], label="Train loss")
    axs[i][0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    axs[i][1].plot(history.epoch, history.history["my_iou_metric_2"], label="Train iou")
    axs[i][1].plot(history.epoch, history.history["val_my_iou_metric_2"], label="Validation iou")
plt.show()
custom = {'my_iou_metric_2': my_iou_metric_2,'lovasz_loss':lovasz_loss}
preds_valid,y_valid = predit_with_one_fold(son_model,x_valid,y_valid,custom)
iou_best,threshold_best = thresholds_select(preds_valid,y_valid)
print(threshold_best)