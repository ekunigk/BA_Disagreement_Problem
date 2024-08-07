import pickle
import torch

from os import listdir
from os.path import isfile, join

class DataCollector():

    def __init(self):
        pass

    def collect_data(self, dataset_name):

        if dataset_name not in ['breastw', 'btsc', 'spambase', 'spf']:
            raise ValueError('Invalid dataset name')

        path = path = f'data/explanation_sets/{dataset_name}/explanations/'

        files = [f for f in listdir(path) if isfile(join(path, f))]
        explanation_data = {}

        for file in files:
            method = file[-6:-4]
            if method == 'gi':
                method = 'grdpgi'
            with open(path + file, 'rb') as f:
                data_temp = pickle.load(f)
                explanation_data[file] = data_temp[method]

        return explanation_data


    def collect_all_data(self):
        breastw = self.collect_data('breastw')
        btsc = self.collect_data('btsc')
        spambase = self.collect_data('spambase')
        spf = self.collect_data('spf')

        all_explanation_data = {'breastw': breastw, 'btsc': btsc, 'spambase': spambase, 'spf': spf}
        
        return all_explanation_data


    def collect_meta_data(self, dataset_name):
        if dataset_name == 'breastw':
            path = 'data/explanation_sets/breastw/meta_data_pid2956_breastw_11880.pkl'
        elif dataset_name == 'btsc':
            path = 'data/explanation_sets/btsc/meta_data_pid2955_btsc_11880.pkl'
        elif dataset_name == 'spambase':
            path = 'data/explanation_sets/spambase/meta_data_pid2957_spambase_11880.pkl'
        elif dataset_name == 'spf':
            path = 'data/explanation_sets/spf/meta_data_pid2954_spf_11880.pkl'
        else:
            raise ValueError('Invalid dataset name')    

        with open(path, 'rb') as f:
            meta_data = pickle.load(f)
        
        return meta_data



    def collect_original_dataset(self, dataset_name):
        meta_data = self.collect_meta_data(dataset_name)
        X = meta_data['X']
        y = meta_data['Y']

        return X, y
    

    def collect_regression_data(self, explanation_set, method1='ig', method2='ks', model_number=1):

        keys = self.get_keys(explanation_set, model_number)

        # if model_number == 1:
        #     keys = list(explanation_set.keys())[1:6]
        # elif model_number == 2:
        #     keys = list(explanation_set.keys())[7:12]
        # elif model_number == 3:
        #     keys = list(explanation_set.keys())[13:18]

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
    

    def get_keys(self, explanation_set, model_number=1):
        if model_number == 1:
            keys = list(explanation_set.keys())[1:6]
        elif model_number == 2:
            keys = list(explanation_set.keys())[7:12]
        elif model_number == 3:
            keys = list(explanation_set.keys())[13:18]
        
        return keys
    

    def collect_pgi(self, dataset_name):
        if dataset_name not in ['breastw', 'btsc', 'spambase', 'spf']:
            raise ValueError('Invalid dataset name')
        
        path = path = f'data/explanation_sets/{dataset_name}/explanations/'

        files = [f for f in listdir(path) if isfile(join(path, f))]
        pgi_data = {}

        for file in files:
            method = file[-6:-4]
            if method == 'gi':
                method = 'grdpgi'
            with open(path + file, 'rb') as f:
                data_temp = pickle.load(f)
                pgi_data[file] = data_temp['pgile0.33']

        return pgi_data
    

    def collect_file(self, dataset_name):
        if dataset_name not in ['breastw', 'btsc', 'spambase', 'spf']:
            raise ValueError('Invalid dataset name')
        
        path = path = f'data/explanation_sets/{dataset_name}/explanations/'

        files = [f for f in listdir(path) if isfile(join(path, f))]
        explanations = {}

        for file in files:
            method = file[-6:-4]
            if method == 'gi':
                method = 'grdpgi'
            with open(path + file, 'rb') as f:
                data_temp = pickle.load(f)
                explanations[file] = data_temp

        return explanations


        