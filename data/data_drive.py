import numpy as np
import pickle
import pandas as pd
from sklearn import preprocessing

data = pd.read_csv("Sensorless_drive_diagnosis.txt", header=None, sep=" ")
data = data.values
print(data[:2,:])
np.random.shuffle(data)

y = data[:, -1:].astype(np.int) - 1
x = data[:, :-1]

print(y[:20])

scaler = preprocessing.MinMaxScaler()
x_numpy = scaler.fit_transform(x)

with open("drive_x", "wb") as file:
    pickle.dump(x_numpy, file)

with open("drive_y", "wb") as file:
    pickle.dump(y, file)
