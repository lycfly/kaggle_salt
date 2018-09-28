from util import *
from loss import *
# Source https://www.kaggle.com/bguberfain/unet-with-depth
def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs

save_path = '../trained_models/single_res34_dw.model'
train_df = pd.read_csv("/home/zhangs/lyc/salt/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("/home/zhangs/lyc/salt/depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]#将生成id不在train中的样本id集合
x_test = np.array([reflect_pad(np.array(load_img("/home/zhangs/lyc/salt/test/images/{}.png".format(idx),
    grayscale=True))) / 255 for idx in tqdm(test_df.index)]).reshape([-1,256,256,1])
print(x_test.shape)

custom = {'my_iou_metric': my_iou_metric,'bce_dice_loss':bce_dice_loss}
preds_test = predit_with_one_fold_test(save_path,x_test,custom)
threshold_best = 0.5102040816326531
pred_dict = {idx: RLenc(np.round(downsample(preds_test[i]) >
            threshold_best)) for i, idx in enumerate(tqdm(test_df.index.values))}

sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('single_res34_dw_816LB_submission.csv')