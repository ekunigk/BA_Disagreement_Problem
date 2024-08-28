import pickle
import pandas as pd

"""
saving and loading results from time-consuming computations
"""

def save_dict(dictionary, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)


def load_dict(filename):
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
        
    return b
