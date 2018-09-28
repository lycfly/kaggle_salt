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
# print(np.array(x_train).shape)
# #test 数据集划分是否正确
# #plot_flods_coverage(train_df,cov_train,flods_num=5,mode='train',is_single_test=True)
# #plot_flods_coverage(train_df,cov_test,flods_num=5,mode='test',is_single_test=True)
# print('SET split sucessful！')

# 数据增强
# x_train, y_train = flip(x_train,y_train)
# print('Data augument sucessful！')
# 模型训练
save_path = '../trained_models/single_res34_dw.model'

custom = {'my_iou_metric': my_iou_metric,'bce_dice_loss':bce_dice_loss}
preds_valid,y_valid = predit_with_one_fold(save_path,x_valid,y_valid,custom)
iou_best,threshold_best = thresholds_select(preds_valid,y_valid)
print(threshold_best)