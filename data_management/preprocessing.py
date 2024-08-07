from sklearn.preprocessing import MaxAbsScaler
import numpy as np
import torch

def scale_data(data, with_label=True):
    scaled_data = data.clone()
    scaler = MaxAbsScaler()
    if with_label:
        scaled_data[:, :-1] = scaler.fit_transform(scaled_data[:, :-1].T).T
    else:
        scaled_data = scaler.fit_transform(scaled_data.T).T
    return torch.Tensor(scaled_data)


def mask_features(explanation_set, mask=0, k=3):

    number_of_features = len(explanation_set[0])

    number_of_masks = number_of_features - k

    explanation_masked = explanation_set.clone()

    for explanation in explanation_masked:
        explanation_absolute = torch.abs(explanation)
        values, indices = torch.topk(explanation_absolute, number_of_masks, largest=False)
        explanation[indices] = mask
         
    return explanation_masked

