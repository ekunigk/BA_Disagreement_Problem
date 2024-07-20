import numpy as np
from sklearn.model_selection import KFold
import torch


class Translator():

    def __init__(self, model):
        self.model = model

    def translate_kfold(self, explanations1, explanations2):
        X = explanations1
        y = explanations2
        kf = KFold(n_splits=10, random_state=44, shuffle=True)
        mse_scores = []
        for train_idx, test_idx in kf.split(X):
            X_train = torch.index_select(X, 0, torch.tensor(train_idx)).numpy()
            X_test = torch.index_select(X, 0, torch.tensor(test_idx)).numpy()
            y_train, y_test = y[train_idx].numpy(), y[test_idx].numpy()

            self.model.fit(X_train, y_train)
            score = self.model.score(X_test, y_test)
        