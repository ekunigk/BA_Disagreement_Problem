import pickle
import torch

from os import listdir
from os.path import isfile, join

from sklearn.preprocessing import MaxAbsScaler


class DataCollector():

    def __init__(self, explanation_set):
        self.explanation_set_name = explanation_set
        self.explanation_set = self.collect_data(explanation_set)
        self.explanations_all = self.collect_all_methods()
        self.scaled_explanations = self.scale_data(with_label=True)
        self.masked_explanations = self.collect_all_methods()
        

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
    

    def collect_meta_data(self):
        if self.explanation_set_name == 'breastw':
            path = 'data/explanation_sets/breastw/meta_data_pid2956_breastw_11880.pkl'
        elif self.explanation_set_name == 'btsc':
            path = 'data/explanation_sets/btsc/meta_data_pid2955_btsc_11880.pkl'
        elif self.explanation_set_name == 'spambase':
            path = 'data/explanation_sets/spambase/meta_data_pid2957_spambase_11880.pkl'
        elif self.explanation_set_name == 'spf':
            path = 'data/explanation_sets/spf/meta_data_pid2954_spf_11880.pkl'
        else:
            raise ValueError('Invalid dataset name')    

        with open(path, 'rb') as f:
            meta_data = pickle.load(f)
        
        return meta_data
    

    def collect_original_dataset(self):
        meta_data = self.collect_meta_data()
        X = meta_data['X']
        y = meta_data['Y']

        return X, y
    

    def get_keys(self, model_number=1):
        if model_number == 1:
            keys = list(self.explanation_set.keys())[1:6]
        elif model_number == 2:
            keys = list(self.explanation_set.keys())[7:12]
        elif model_number == 3:
            keys = list(self.explanation_set.keys())[13:18]
        
        return keys
    

    def collect_regression_data(self, method1='ig', method2='ks', model_number=1):

        keys = self.get_keys(model_number)

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

        dataset = torch.ones((1, len(self.explanation_set[method_keys[0]][0])+1))

        for i in range(2):
            explanations = self.explanation_set[method_keys[i]]
            data = torch.hstack((explanations, torch.full((len(explanations), 1), i)))
            dataset = torch.vstack((dataset, data))
        dataset = dataset[1:]

        return dataset
    

    def collect_all_methods(self, model_number=1):
        keys = self.get_keys(model_number)

        dataset = torch.ones((1, len(self.explanation_set[keys[0]][0])+1))

        for i in range(len(keys)):
            explanations = self.explanation_set[keys[i]]
            data = torch.hstack((explanations, torch.full((len(explanations), 1), i)))
            dataset = torch.vstack((dataset, data))

        dataset = dataset[1:]

        return dataset
        

    def collect_pgi(self):
        print(self.explanation_set_name)
        if self.explanation_set_name not in ['breastw', 'btsc', 'spambase', 'spf']:
            raise ValueError('Invalid dataset name')
        
        path = path = f'data/explanation_sets/{self.explanation_set_name}/explanations/'

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
    

    def collect_file(self):
        if self.explanation_set_name not in ['breastw', 'btsc', 'spambase', 'spf']:
            raise ValueError('Invalid dataset name')
        
        path = path = f'data/explanation_sets/{self.explanation_set_name}/explanations/'

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
    

    def scale_data(self, with_label=True, model_number=1):
        explanations_all = self.collect_all_methods(model_number=model_number)
        scaled_data = explanations_all.clone()
        scaler = MaxAbsScaler()
        if with_label:
            scaled_data[:, :-1] = torch.Tensor(scaler.fit_transform(scaled_data[:, :-1].T).T)
        else:
            scaled_data = torch.Tensor(scaler.fit_transform(scaled_data.T).T)
        return scaled_data
    

    def mask_features(self, k=3, mask=0, scaled=False):
        
        number_of_features = len(self.explanations_all[0])-1

        number_of_masks = number_of_features - k

        if scaled:
            explanation_masked = self.scaled_explanations.clone()
        else:
            explanation_masked = self.explanations_all.clone()

        for explanation in explanation_masked[:, :-1]:
            explanation_absolute = torch.abs(explanation)
            values, indices = torch.topk(explanation_absolute, number_of_masks, largest=False)
            explanation[indices] = mask

        self.masked_explanations[:, :-1] = explanation_masked[:, :-1]
            
        return explanation_masked