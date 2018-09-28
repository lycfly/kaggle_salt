from util import *
from loss import *
import time
gpu_control('0',0.9)

def rle_encode(im):
    pixels = im.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

save_path = '../trained_models/single_res34_dw.model'
train_df = pd.read_csv("/home/zhangs/lyc/salt/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("/home/zhangs/lyc/salt/depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]#将生成id不在train中的样本id集合
custom = {'my_iou_metric': my_iou_metric, 'bce_dice_loss': bce_dice_loss}
model = load_model(save_path, custom_objects=custom)

batch_size = 500
preds_test = []
i = 0
while i < test_df.shape[0]:
    print('Images Processed:', i)
    index_val = test_df.index[i:i + batch_size]
    depth_val = test_df.z[i:i + batch_size]
    x_test = np.array([reflect_pad(np.array(load_img("/home/zhangs/lyc/salt/test/images/{}.png".format(idx),
                             grayscale=True))) / 255 for idx in tqdm(index_val)]).reshape([-1, 256, 256, 1])
    #x_test = np.array([upsample(np.array(load_img("test/images/{}.png".format(idx), grayscale=True))) / 255 for idx in
    #                   (index_val)]).reshape(-1, img_size_target, img_size_target, 1)
    preds_test_temp = predit_with_one_fold_test(model, x_test)
    if i == 0:
        preds_test = preds_test_temp
    else:
        preds_test = np.concatenate([preds_test, preds_test_temp], axis=0)
    #     print(preds_test.shape)
    i += batch_size
print('test size：'+str(preds_test.shape))
threshold_best =0.55

t1 = time.time()
pred_dict = {idx: rle_encode(np.round(downsample(preds_test[i]) > threshold_best)) for i, idx in
             enumerate(tqdm(test_df.index.values))}
t2 = time.time()
print('Usedtime = '+str(t2-t1)+' s')
sub = pd.DataFrame.from_dict(pred_dict, orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('the055_single_res34_dw_816LB_submission.csv')

