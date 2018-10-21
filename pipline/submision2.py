from util import *
from loss import *
import time
gpu_control('1',0.5)

def rle_encode(im):
    pixels = im.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

save_path = '/home/zhangs/lyc/salt/trained_models/2018.10.9/flod5_single_lovasz_adam.model'
train_df = pd.read_csv("/home/zhangs/lyc/salt/data/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("/home/zhangs/lyc/salt/data/depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]#将生成id不在train中的样本id集合
custom = {'my_iou_metric_2': my_iou_metric_2, 'lovasz_loss': lovasz_loss}
model = load_model(save_path, custom_objects=custom)

batch_size = 500
im_shape = 101
preds_test = []
i = 0
multiply_model = True
if multiply_model:
    model0 = load_model(
        '/home/zhangs/lyc/salt/trained_models/2018.10.9/flod%d_single_lovasz_adam.model'%0, custom_objects=custom)
    model1 = load_model(
        '/home/zhangs/lyc/salt/trained_models/2018.10.9/flod%d_single_lovasz_adam.model'%1, custom_objects=custom)
    model2 = load_model(
        '/home/zhangs/lyc/salt/trained_models/2018.10.9/flod%d_single_lovasz_adam.model'%2, custom_objects=custom)
    model3 = load_model(
        '/home/zhangs/lyc/salt/trained_models/2018.10.9/flod%d_single_lovasz_adam.model' % 3, custom_objects=custom)
    model4 = load_model(
        '/home/zhangs/lyc/salt/trained_models/2018.10.9/flod%d_single_lovasz_adam.model' % 4, custom_objects=custom)
else:
    model0 =None
    model1 =None
    model2 =None
    model3 =None
    model4 =None
while i < test_df.shape[0]:
    print('Images Processed:', i)
    index_val = test_df.index[i:i + batch_size]
    depth_val = test_df.z[i:i + batch_size]
    x_test = np.array([np.array(load_img("/home/zhangs/lyc/salt/data/test/images/{}.png".format(idx),
                             grayscale=True)) / 255 for idx in tqdm(index_val)]).reshape([-1, im_shape, im_shape, 1])
    #x_test = np.array([upsample(np.array(load_img("test/images/{}.png".format(idx), grayscale=True))) / 255 for idx in
    #                   (index_val)]).reshape(-1, img_size_target, img_size_target, 1)
    if multiply_model:
        preds_test_temp0 = predit_with_one_fold_raw(model, x_test,None,custom,is_test=True)
        preds_test_temp1 = predit_with_one_fold_raw(model0, x_test,None,custom,is_test=True)
        preds_test_temp2 = predit_with_one_fold_raw(model1, x_test,None,custom,is_test=True)
        preds_test_temp3 = predit_with_one_fold_raw(model2, x_test,None,custom,is_test=True)
        preds_test_temp4 = predit_with_one_fold_raw(model3, x_test,None,custom,is_test=True)
        preds_test_temp5 = predit_with_one_fold_raw(model4, x_test,None,custom,is_test=True)

        preds_test_temp = preds_test_temp0+preds_test_temp1+preds_test_temp2+preds_test_temp3+preds_test_temp4+preds_test_temp5
        preds_test_temp = preds_test_temp/6
    else:
        preds_test_temp = predit_with_one_fold_raw(model, x_test,None,custom,is_test=True)
    if i == 0:
        preds_test = preds_test_temp
    else:
        preds_test = np.concatenate([preds_test, preds_test_temp], axis=0)
    #     print(preds_test.shape)
    i += batch_size
print('test size：'+str(preds_test.shape))
threshold_best =0.0

t1 = time.time()
pred_dict = {idx: rle_encode(np.round(downsample(preds_test[i]) > threshold_best)) for i, idx in
             enumerate(tqdm(test_df.index.values))}
t2 = time.time()
print('Usedtime = '+str(t2-t1)+' s')
sub = pd.DataFrame.from_dict(pred_dict, orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('single-small-6merge+0.csv')

