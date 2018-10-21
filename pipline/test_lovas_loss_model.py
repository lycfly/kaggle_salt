from util import *
from loss import *
from augument import *
from segmentation_models.segmentation_models import Unet
from segmentation_models.segmentation_models.utils import set_trainable
from keras.backend.tensorflow_backend import set_session
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))

# 数据导入
train_df = data_initial()

# 构造k折交叉验证
K_flods = 6
x_train,x_valid,y_train,y_valid,cov_train,cov_test,depth_train,depth_test =\
                             k_folds(train_df,K_flods,padd=edge_pad,is_single_test=False)

# test 数据集划分是否正确
# plot_flods_coverage(train_df,cov_train,flods_num=6,mode='train',is_single_test=False)
# plot_flods_coverage(train_df,cov_test,flods_num=6,mode='test',is_single_test=False)
print('SET split sucessful！')


# 模型训练
history_all = []
num_of_flods =K_flods

metric = my_iou_metric_2
metric_str = 'my_iou_metric_2'

custom = {metric_str: metric,'lovasz_loss':lovasz_loss}
fig, axs = plt.subplots(num_of_flods, 2, squeeze=False, figsize=(15, 5))
save_path = '/home/zhangs/lyc/salt/trained_models/flod2_single_res34_edge_sgd_focal_loss_son.model'

x_valid_flods, y_valid_flods = x_valid[2],y_valid[2]
preds_valid, y_valid_true = predit_with_one_fold(save_path, x_valid_flods, y_valid_flods, custom)
iou_best, threshold_best = thresholds_select_lovas(preds_valid, y_valid_true)
print([iou_best,threshold_best])

