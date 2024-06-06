import torch 

def collect_regression_data(explanation_set, method1='ig', method2='ks', model_number=1):
    if model_number == 1:
        keys = list(explanation_set.keys())[1:6]
    elif model_number == 2:
        keys = list(explanation_set.keys())[7:12]
    elif model_number == 3:
        keys = list(explanation_set.keys())[13:18]

    method_keys = []
    for i in range(2):
        method = [method1, method2][i]
        if method == 'ig':
            method_keys.append(keys[0])
        elif method == 'ks':
            method_keys.append(keys[1])
        elif method == 'li':
            method_keys.append(keys[2])
        elif method == 'sg':
            method_keys.append(keys[3])
        elif method == 'vg':
            method_keys.append(keys[4])

    dataset = torch.ones((1, len(explanation_set[method_keys[0]][0])+1))

    for i in range(2):
        explanations = explanation_set[method_keys[i]]
        data = torch.hstack((explanations, torch.full((len(explanations), 1), i)))
        dataset = torch.vstack((dataset, data))
    dataset = dataset[1:]

    return dataset


def prepare_umap_data(explanation_set, model_number=1):
    if model_number == 1:
        keys = list(explanation_set.keys())[1:6]
    elif model_number == 2:
        keys = list(explanation_set.keys())[7:12]
    elif model_number == 3:
        keys = list(explanation_set.keys())[13:18]

    dataset = torch.ones((1, len(explanation_set[keys[0]][0])+1))

    for i in range(len(keys)):
        explanations = explanation_set[keys[i]]
        data = torch.hstack((explanations, torch.full((len(explanations), 1), i)))
        dataset = torch.vstack((dataset, data))

    dataset = dataset[1:]

    return dataset