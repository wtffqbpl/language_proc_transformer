#! coding: utf-8

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import utils.dlf as dlf
from utils.plot import plot

dlf.DATA_HUB['kaggle_house_train'] = (
    dlf.DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

dlf.DATA_HUB['kaggle_house_test'] = (
    dlf.DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')


class HousePricePrediction:
    def __init__(self):
        self.train_data = pd.read_csv(dlf.download('kaggle_house_train'))
        self.test_data = pd.read_csv(dlf.download('kaggle_house_test'))

        # print(self.train_data.shape)
        # print(self.test_data.shape)
        # print(self.train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

        # Remove id from training and testing datasets, and remove prices from training dataset.
        # The all_features contains training and testing datasets without id and prices.
        all_features = pd.concat(
            (self.train_data.iloc[:, 1:-1], self.test_data.iloc[:, 1:]))

        # Standardize the data by rescaling features to zero mean and unit variance
        # Intuitively, we standardize the data for two reasons.First, it proves
        # convenient for optimization.Second, because we do not know a priori
        # which features will be relevant, we do not want to penalize coefficients
        # assigned to one feature more than any other.
        numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
        all_features[numeric_features] = all_features[numeric_features].apply(
            lambda x: (x - x.mean()) / (x.std()))
        # Replace NAN numerical features by 0
        all_features[numeric_features] = all_features[numeric_features].fillna(0)
        print(all_features.shape)

        # Now we process the discrete values using One-Hot encoding
        all_features = pd.get_dummies(all_features, dummy_na=True)
        print(all_features.shape)

        n_train = self.train_data.shape[0]
        print(n_train)
        self.train_features = torch.from_numpy(all_features[:n_train].values.astype(float)).to(dtype=torch.float32)
        test_features = torch.from_numpy(all_features[n_train:].values.astype(float)).to(dtype=torch.float32)
        self.train_labels = torch.from_numpy(self.train_data.SalePrice.values.reshape(-1, 1)).to(dtype=torch.float32)

        self.loss = nn.MSELoss()
        self.in_features = self.train_features.shape[1]

    def get_net(self):
        net = nn.Sequential(nn.Linear(self.in_features, 1))
        return net

    def log_rmse(self, net, loss, features, labels):
        # Clamps all elements in input features the range [ 1, inf ].
        clipped_preds = torch.clamp(net(features), 1, float('inf'))
        rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
        return rmse.item()

    def load_array(self, data_arrays, batch_size, is_train=True):
        dataset = data.TensorDataset(*data_arrays)
        return data.DataLoader(dataset, batch_size, shuffle=is_train)

    def train(self, net, loss, train_features, train_labels, test_features, test_labels,
              num_epochs, learning_rate, weight_decay, batch_size):
        train_ls, test_ls = [], []
        train_iter = self.load_array((train_features, train_labels), batch_size)

        optimizer = torch.optim.Adam(net.parameters(),
                                     lr=learning_rate,
                                     weight_decay=weight_decay)

        for epoch in range(num_epochs):
            for x, y in train_iter:
                optimizer.zero_grad()
                l = loss(net(x), y)
                l.backward()
                optimizer.step()
            train_ls.append(self.log_rmse(net, loss, train_features, train_labels))
            if test_labels is not None:
                test_ls.append(self.log_rmse(net, loss, test_features, test_labels))

        return train_ls, test_ls

    def get_k_fold_data(self, k, i, x, y):
        assert k > 1
        fold_size = x.shape[0] // k
        x_train, y_train, x_valid, y_valid = None, None, None, None
        for j in range(k):
            idx = slice(j * fold_size, (j + 1) * fold_size)
            x_part, y_part = x[idx, :], y[idx]

            if j == i:
                x_valid, y_valid = x_part, y_part
            elif x_train is None:
                x_train, y_train = x_part, y_part
            else:
                x_train = torch.cat([x_train, x_part], 0)
                y_train = torch.cat([y_train, y_part], 0)

        return x_train, y_train, x_valid, y_valid

    def k_fold(self, k, x_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
        train_l_sum, valid_l_sum = 0, 0
        for i in range(k):
            k_data = self.get_k_fold_data(k, i, x_train, y_train)
            net = self.get_net()
            train_ls, valid_ls = self.train(net, self.loss, *k_data, num_epochs, learning_rate, weight_decay, batch_size)
            train_l_sum += train_ls[-1]
            valid_l_sum += valid_ls[-1]

            if i == 0:
                plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')

            print(f'The {i + 1}-th traing loss rmse{float(train_ls[-1]):f}, ',
                  f'validation log rmse{float(valid_ls[-1]):f}')

        return train_l_sum / k, valid_l_sum / k

    def train_wrapper(self):
        k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
        train_l, valid_l = self.k_fold(k, self.train_features, self.train_labels, num_epochs, lr,
                                       weight_decay, batch_size)
        print(f'{k}-fold validation: average traing log rmse: {float(train_l):f}, ',
              f'average testing log rmse: {float(valid_l):f}')


if __name__ == "__main__":
    model = HousePricePrediction()

    model.train_wrapper()

    pass
