"""
For more information about MissForest see:
https://academic.oup.com/bioinformatics/article/28/1/112/219101
"""
from missingpy import MissForest
import numpy as np
from utils import load_data
from utils import mse as mse_own


def main(p_miss=0.5, dataset="drive", mode="mcar", para=0.5, train=None, rand_seed=42):
    np.random.seed(rand_seed)

    n, p, xmiss, xhat_0, mask, data_x, data_y = load_data(p_miss, dataset=dataset, mode=mode, para=para, train=train, rand_seed=rand_seed)

    imputer = MissForest(decreasing=True, random_state=rand_seed, verbose=True)
    x_filled = imputer.fit_transform(xmiss)

    mse = mse_own(x_filled, data_x, mask)

    print("MSE for MissForest: ", mse)

    return x_filled, mse

if __name__ == "__main__":
    main()
