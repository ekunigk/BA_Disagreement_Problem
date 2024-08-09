import pickle
import torch

from os import listdir
from os.path import isfile, join

from sklearn.preprocessing import MaxAbsScaler


class DataCollector():

    def __init__(self, explanation_set, model_number=1):
        self.explanation_set_name = explanation_set
        self.explanation_set = self.collect_data(explanation_set)
        self.explanations_all = self.collect_all_methods(model_number=model_number)
        self.scaled_explanations = self.scale_data(with_label=True, model_number=model_number)
        self.masked_explanations = self.collect_all_methods(model_number=model_number)
        self.explanation_method_length = int(len(self.explanations_all) / 5)
        self.non_zero_explanations = self.create_non_zero_dataset()
        self.non_zero_masked_explanations = self.non_zero_explanations.clone()
        

    def collect_data(self, dataset_name):

        if dataset_name not in ['breastw', 'btsc', 'spambase', 'spf']:
            raise ValueError('Invalid dataset name')

        path = path = f'data_management/explanation_sets/{dataset_name}/explanations/'

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
            path = 'data_management/explanation_sets/breastw/meta_data_pid2956_breastw_11880.pkl'
        elif self.explanation_set_name == 'btsc':
            path = 'data_management/explanation_sets/btsc/meta_data_pid2955_btsc_11880.pkl'
        elif self.explanation_set_name == 'spambase':
            path = 'data_management/explanation_sets/spambase/meta_data_pid2957_spambase_11880.pkl'
        elif self.explanation_set_name == 'spf':
            path = 'data_management/explanation_sets/spf/meta_data_pid2954_spf_11880.pkl'
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
        lime_included = False
        for i in range(2):
            method = [method1, method2][i]
            if method == 'ig':
                method_keys.append(keys[0])
            elif method == 'ks':
                method_keys.append(keys[1])
            elif method == 'li':
                method_keys.append(keys[2])
                lime_included = True
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
        
        path = path = f'data_management/explanation_sets/{self.explanation_set_name}/explanations/'

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
        
        path = path = f'data_management/explanation_sets/{self.explanation_set_name}/explanations/'

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
    

    def mask_features(self, k=3, mask=0, scaled=True):
        
        number_of_features = len(self.explanations_all[0])-1

        number_of_masks = number_of_features - k

        masked_indices = torch.zeros((1, number_of_masks))
        masked_indices_non_zero = torch.zeros((1, number_of_masks))

        if scaled:
            explanation_masked = self.scaled_explanations.clone()
            non_zero_explanations = self.non_zero_explanations.clone()
        else:
            explanation_masked = self.explanations_all.clone()
            non_zero_explanations = self.non_zero_masked_explanations.clone()

        for explanation in explanation_masked[:, :-1]:
            explanation_absolute = torch.abs(explanation)
            values, indices = torch.topk(explanation, number_of_masks, largest=False)
            explanation[indices] = mask
            masked_indices = torch.vstack((masked_indices, indices))

        self.masked_explanations[:, :-1] = explanation_masked[:, :-1]

        for explanation in non_zero_explanations[:, :-1]:
            explanation_absolute = torch.abs(explanation)
            values, indices = torch.topk(explanation, number_of_masks, largest=False)
            explanation[indices] = mask
            masked_indices_non_zero = torch.vstack((masked_indices_non_zero, indices))

        self.non_zero_masked_explanations[:, :-1] = non_zero_explanations[:, :-1]

        return masked_indices[1:], masked_indices_non_zero[1:]

    

    def find_lime_zero_explanations(self):
        explanation_length = self.explanation_method_length
        index_list = []
        explanation_count = 0
        for explanation in self.scaled_explanations[2*explanation_length:3*explanation_length, :-1]:
            non_zero_features = 0
            for i in range(len(explanation)):
                if explanation[i] != 0:
                    non_zero_features += 1
            if non_zero_features == 0:
                index_list.append(explanation_count)
            explanation_count += 1
        return index_list
    

    def create_non_zero_dataset(self):
        index_list = self.find_lime_zero_explanations()
        explanation_length = self.explanation_method_length
        lime_explanation_set = self.scaled_explanations.clone()
        remaining_indices = [i for i in range(explanation_length) if i not in index_list]
        all_indices = []
        for i in range(5):
            all_indices += [i*explanation_length + index for index in remaining_indices]
        lime_explanation_set = lime_explanation_set[all_indices]
        return lime_explanation_set
            
