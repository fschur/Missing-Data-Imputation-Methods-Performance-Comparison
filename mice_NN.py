# This code is adapted from https://github.com/Ouwen/scikit-mice/blob/master/skmice.py

from utils import load_data, mse
from sklearn.preprocessing import Imputer

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import numpy as np


class MiceImputer(object):

    def __init__(self, missing_values="NaN", strategy="mean", axis=0, verbose=0, copy=True):
        self.missing_values = missing_values
        self.strategy = strategy
        self.axis = axis
        self.verbose = verbose
        self.copy = copy
        self.imp = Imputer(missing_values=self.missing_values, strategy=self.strategy, axis=self.axis,
                           verbose=self.verbose, copy=self.copy)

    def _seed_values(self, X):
        self.imp.fit(X)
        return self.imp.transform(X)

    def _get_mask(self, X, value_to_mask):
        if value_to_mask == "NaN" or np.isnan(value_to_mask):
            return np.isnan(X)
        else:
            return np.array(X == value_to_mask)

    def _process(self, X, column, sizes, activation, epochs, mask, lr):
        # Remove values that are in mask
        mask_col = mask[:, column]
        #!!mask = np.array(self._get_mask(X)[:, column].T)[0]
        mask_indices = np.where(mask_col == True)[0]
        X_data = np.delete(X, mask_indices, 0)

        # Instantiate the model
        model = MLPRegressor(hidden_layer_sizes=sizes, activation=activation,
                             solver='adam', learning_rate_init=lr,max_iter=epochs)

        # Slice out the column to predict and delete the column.
        y_data = X_data[:, column]
        X_data = np.delete(X_data, column, 1)

        # Split training and test data
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=42)

        # Fit the model
        model.fit(X_train, y_train)

        # Score the model
        scores = model.score(X_test, y_test)

        # Predict missing vars
        X_predict = np.delete(X, column, 1)
        y = model.predict(X_predict)

        # Replace values in X with their predictions
        X[mask_indices, column] = np.take(y, mask_indices)
        #np.put(X, predict_indicies, np.take(y, mask_col))
        # Return model and scores
        return X, scores

    def transform(self, X, sizes, activation='relu', epochs=500, iterations=10, lr=0.001):
        X = np.array(X)
        mask = self._get_mask(X, self.missing_values)
        X = self._seed_values(X)
        specs = np.zeros((iterations, X.shape[1]))

        for i in range(iterations):
            print(i)
            for c in range(X.shape[1]):
                X, specs[i][c] = self._process(X, c, sizes, activation, epochs, mask, lr)

        # Return X matrix with imputed values
        return X, specs


def main(p_miss=0.5, hidden_size=100, epochs=70, lr=0.001,
         dataset="drive", mode="mcar", para=0.5, train=None):

    n, p, xmiss, xhat_0, mask, data_x, data_y = load_data(p_miss, dataset=dataset, mode=mode, para=para, train=train)

    imputer = MiceImputer(np.nan)
    X = xmiss

    X_filled, specs = imputer.transform(np.array(X), (hidden_size, hidden_size, hidden_size),
                                        epochs=epochs, lr=lr, iterations=10)

    mse_nn = mse(X_filled, data_x, mask)
    print("MSE MICE_NN : ", mse_nn)

    return X_filled, mse_nn

if __name__ == "__main__":
    main()
