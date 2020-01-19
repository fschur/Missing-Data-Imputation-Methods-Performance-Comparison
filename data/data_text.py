import numpy as np
import pickle
import pandas as pd
from sklearn import preprocessing

data = pd.read_csv("page-blocks.data", header=None, sep="[ ]+")
data = data.values

np.random.shuffle(data)

y = data[:, -1:].astype(np.int) - 1
x = data[:, :-1]

scaler = preprocessing.MinMaxScaler()
x_numpy = scaler.fit_transform(x)

with open("text_x", "wb") as file:
    pickle.dump(x_numpy, file)

with open("text_y", "wb") as file:
    pickle.dump(y, file)
