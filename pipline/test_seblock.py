from util import *
from loss import *
from augument import *
from keras.backend.tensorflow_backend import set_session
from se_resnet import SEResNet34

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))

# 数据导入
train_df = data_initial()

# 构造k折交叉验证
K_flods = 6
x_train,x_valid,y_train,y_valid,cov_train,cov_test,depth_train,depth_test =\
                             k_folds(train_df,K_flods,padd=edge_pad,is_single_test=True)
print(np.array(x_train).shape)
#test 数据集划分是否正确
#plot_flods_coverage(train_df,cov_train,flods_num=5,mode='train',is_single_test=True)
#plot_flods_coverage(train_df,cov_test,flods_num=5,mode='test',is_single_test=True)
print('SET split sucessful！')

# 数据增强
x_train, y_train = flip(x_train,y_train)
print('Data augument sucessful！')
# 模型训练
history_all = []
num_of_flods =1
#(np.repeat(x_train[..., :1], 3, axis=-1))


model =SEResNet34(input_shape=(256, 256, 3),
           width=1,
           bottleneck=False,
           weight_decay=1e-4,
           include_top=False,
           weights='imagenet',
           pooling=None,
           classes=1)
model.summary()
