import numpy as np
def flip(x_train,y_train):
    x_l_r_flip = [np.fliplr(x) for x in x_train]
    y_l_r_flip = [np.fliplr(x) for x in y_train]

    x_train = np.append(x_train, x_l_r_flip, axis=0)
    y_train = np.append(y_train, y_l_r_flip, axis=0)
    return x_train,y_train