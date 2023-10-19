import pickle
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader


def sample(subj_data_dict):
    


class ZuCo_dataset(Dataset):
    def __init__(self, input_datasets_list, tokenizer, phase, eeg_type='GD',
                 bands = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'],
                 is_add_CLS_token=False):
        
        self.tokenizer = tokenizer

        if not isinstance(input_datasets_list, list):
            if not isinstance(input_datasets_list, dict):
                raise ValueError('input_datasets_list should be list of dict, but got type "{}"'.format(type(input_datasets_list)))
            input_datasets_list = [input_datasets_list]
        
        for input_data_dict in input_datasets_list:
            subjects = list(input_data_dict.keys())
            print('\n[*] Subjects: ', subjects)

            total_num_sentences = len(input_data_dict[subjects[0]])

            # take first 80% as trainset, 10% as dev and 10% as test
            train_len = int(0.8 * total_num_sentences)
            dev_len = train_len + int(0.1 * total_num_sentences)

            print(f'\n[*] Train: {train_len}, Dev: {dev_len}, Test: {total_num_sentences - dev_len}')

            if phase == 'train':



