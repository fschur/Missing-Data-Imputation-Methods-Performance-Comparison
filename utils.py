import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from fancyimpute import SimpleFill

def sigmoid(x, para=0.5):
    s = 1/(1+np.exp(-15*(x-para)))
    return s

def load_data(p_miss, dataset="drive", mode="mcar", para=0.5, train=None, rand_seed=42):
    np.random.seed(rand_seed)

    with open("data/" + dataset + "_x", "rb") as file:
        data_x = pickle.load(file)
    with open("data/" + dataset + "_y", "rb") as file:
        data_y = pickle.load(file)

    n = data_x.shape[0]
    p = data_x.shape[1]

    perc_miss = p_miss
    xmiss = np.copy(data_x)

    if mode == "mcar":
        xmiss_flat = xmiss.flatten()
        miss_pattern = np.random.choice(n*p, np.floor(n*p*perc_miss).astype(np.int), replace=False)
        xmiss_flat[miss_pattern] = np.nan
        xmiss = xmiss_flat.reshape([n, p])  # in xmiss, the missing values are represented by nans
    elif mode == "mar":
        fixed_len = int(np.floor(p/3))
        prob = para*np.mean(data_x[:, :fixed_len], 1)
        prob = sigmoid(prob, 0.5)
        for i in range(n):
            mask_tmp = np.random.choice([1, 0], size=p, p=[1 - prob[i], prob[i]])
            for j in range(fixed_len, p):
                if mask_tmp[j] == 0:
                    xmiss[i, j] = np.nan
        print("missing rate: ", np.sum(np.isnan(xmiss.flatten()))/(n*p))
    else:
        raise Exception("mode is not valid")

    mask = np.isfinite(xmiss) # binary mask that indicates which values are missing

    xhat_0 = np.copy(xmiss)
    xhat_0[np.isnan(xmiss)] = 0

    x_filled = SimpleFill().fit_transform(xmiss)

    print("MSE mean imputation full data: " + str(mse(x_filled, data_x, mask)))

    if train == True:
        part = int(np.floor(n/2))
        return (n-part), p, xmiss[part:,:], xhat_0[part:,:], mask[part:,:], data_x[part:,:], data_y[part:,:]
    elif train == False:
        part = int(np.floor(n/2))
        return part, p, xmiss[:part,:], xhat_0[:part,:], mask[:part,:], data_x[:part,:], data_y[:part,:]
    elif train == None:
        return n, p, xmiss, xhat_0, mask, data_x, data_y

def mse(xhat,xtrue,mask):
    xhat = np.array(xhat)
    xtrue = np.array(xtrue)
    t = np.power(xhat-xtrue, 2)
    return np.mean(t[~mask])
