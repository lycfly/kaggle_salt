from util import *
from loss import *
from augument import *
from segmentation_models.segmentation_models import Unet
from segmentation_models.segmentation_models.utils import set_trainable
from keras.backend.tensorflow_backend import set_session
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
print('SET split sucessful！')
a = edge_pad(x_train[0,:,:,0])
plt.imshow(a)
plt.show()
#(np.repeat(x_train[..., :1], 3, axis=-1))


