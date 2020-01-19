from fancyimpute import IterativeImputer
import numpy as np
from utils import mse as mse_own
from utils import load_data


def main(p_miss=0.5, dataset="drive", mode="mcar", para=0.5, train=None, rand_seed=42):
    np.random.seed(rand_seed)

    n, p, xmiss, xhat_0, mask, data_x, data_y = load_data(p_miss, dataset=dataset, mode=mode, para=para, train=train, rand_seed=rand_seed)

    x_filled = IterativeImputer().fit_transform(xmiss)

    mse = mse_own(x_filled, data_x, mask)

    print("MSE for MICE: " + str(mse))

    return x_filled, mse

if __name__ == "__main__":
    main()