import pickle
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import BartTokenizer
import argparse


def normalize(tensor):
    mean = torch.mean(tensor)
    std = torch.std(tensor)
    return (tensor - mean) / std

def sample(subj_data_dict, tokenizer, eeg_type='GD', 
           bands=['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], add_CLS_token=False, max_len=57):
    
    def get_eeg_data(word_obj, eeg_type, bands):
        eeg_data = []
        for band in bands:
            eeg_data.append(word_obj['word_level_EEG'][eeg_type][eeg_type+band])
        word_eeg_embedding = np.concatenate(eeg_data)
        if len(word_eeg_embedding) != 105*len(bands):
            print(f'!!!!! word_eeg_embedding length should be {105*len(bands)}, but got {len(word_eeg_embedding)}')
            return None
        tensor = torch.from_numpy(word_eeg_embedding)

        return normalize(tensor)
    
    if subj_data_dict is None:
        return None
    
    input_sample = {}
    target_string = subj_data_dict['content']
    target_tokens = tokenizer(target_string, return_tensors='pt', padding='max_length', max_length=max_len,
                              truncation=True, return_attention_mask=True)
    
    input_sample['target_ids'] = target_tokens['input_ids'][0]

    # correct the typos in the dataset
    if 'emp11111ty' in target_string:
        target_string = target_string.replace('emp11111ty','empty')
    if 'film.1' in target_string:
        target_string = target_string.replace('film.1','film.')
    
    word_embeddings = []
    if add_CLS_token:
        word_embeddings.append(torch.zeros(105*len(bands)))

    for word_obj in subj_data_dict['word']:
        word_eeg_embedding_tensor = get_eeg_data(word_obj, eeg_type, bands)
        if word_eeg_embedding_tensor is None:
            return None
        if torch.isnan(word_eeg_embedding_tensor).any():
            return None
        word_embeddings.append(word_eeg_embedding_tensor)

    while len(word_embeddings) < max_len:
        word_embeddings.append(torch.zeros(105*len(bands)))

    input_sample['input_embeddings'] = torch.stack(word_embeddings)

    input_sample['input_attn_mask'] = torch.zeros(max_len)
    if add_CLS_token:
        input_sample['input_attn_mask'][:len(subj_data_dict['word'])+1] = torch.ones(len(subj_data_dict['word'])+1)
    else:
        input_sample['input_attn_mask'][:len(subj_data_dict['word'])] = torch.ones(len(subj_data_dict['word']))

    input_sample['input_attn_mask_invert'] = torch.ones(max_len)
    if add_CLS_token:
        input_sample['input_attn_mask_invert'][:len(subj_data_dict['word'])+1] = torch.zeros(len(subj_data_dict['word'])+1)
    else:
        input_sample['input_attn_mask_invert'][:len(subj_data_dict['word'])] = torch.zeros(len(subj_data_dict['word']))

    # mask out target padding for computing cross entropy loss
    input_sample['target_mask'] = target_tokens['attention_mask'][0]
    input_sample['seq_len'] = len(subj_data_dict['word'])

    if input_sample['seq_len'] == 0:
        print('!!!!! seq_len is 0. Discarding instace...')
        return None
    
    return input_sample

    


class ZuCo_dataset(Dataset):
    def __init__(self, input_datasets_list, tokenizer, phase, eeg_type='GD',
                 bands = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'],
                 add_CLS_token=False):
        self.inputs = []
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
                print('\n[*] Loading trainset...')
                for key in subjects:
                    for i in range(train_len):
                        input_sample = sample(input_data_dict[key][i], self.tokenizer, eeg_type, bands, add_CLS_token)
                        if input_sample is not None:
                            self.inputs.append(input_sample)
            elif phase == 'dev':
                print('\n[*] Loading devset...')
                for key in subjects:
                    for i in range(train_len, dev_len):
                        input_sample = sample(input_data_dict[key][i], self.tokenizer, eeg_type, bands, add_CLS_token)
                        if input_sample is not None:
                            self.inputs.append(input_sample)
            elif phase == 'test':
                print('\n[*] Loading testset...')
                for key in subjects:
                    for i in range(dev_len, total_num_sentences):
                        input_sample = sample(input_data_dict[key][i], self.tokenizer, eeg_type, bands, add_CLS_token)
                        if input_sample is not None:
                            self.inputs.append(input_sample)

            print(f'[*] {phase}set size: {len(self.inputs)}')
        
        print(f'\n[*] Dataset loaded. Input tensor size {self.inputs[0]["input_embeddings"].size()}, target tensor size {self.inputs[0]["target_ids"].size()}')
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_sample = self.inputs[idx]
        return (input_sample['input_embeddings'],
                input_sample['seq_len'],
                input_sample['input_attn_mask'],
                input_sample['input_attn_mask_invert'],
                input_sample['target_ids'],
                input_sample['target_mask'])
        

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eeg-type', type=str, default='GD')
    parser.add_argument('--bands', nargs='+', default=['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'])
    parser.add_argument('--add-CLS-token', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print()
    print('EEG type: ', args.eeg_type)
    print('Bands: ', args.bands)
    
    print('\n[*] Loading dataset...')
    dataset_dicts = []

    dataset_path_task1 = './dataset/processed/Task1_SR_processed.pickle'
    with open(dataset_path_task1, 'rb') as f:
        dataset_dicts.append(pickle.load(f))

    dataset_path_task2 = './dataset/processed/Task2_NR_processed.pickle'
    with open(dataset_path_task2, 'rb') as f:
        dataset_dicts.append(pickle.load(f))
    
    dataset_path_task3 = './dataset/processed/Task3_TSR_processed.pickle'
    with open(dataset_path_task3, 'rb') as f:
        dataset_dicts.append(pickle.load(f))

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    train_args = {'input_datasets_list': dataset_dicts,
                  'phase': 'train',
                  'tokenizer': tokenizer,
                  'eeg_type': args.eeg_type,
                  'bands': args.bands,
                  'add_CLS_token': args.add_CLS_token}
    train_dataset = ZuCo_dataset(**train_args)

    dev_args = {'input_datasets_list': dataset_dicts,
                'phase': 'dev',
                'tokenizer': tokenizer,
                'eeg_type': args.eeg_type,
                'bands': args.bands,
                'add_CLS_token': args.add_CLS_token}
    dev_dataset = ZuCo_dataset(**dev_args)

    test_args = {'input_datasets_list': dataset_dicts,
                 'phase': 'test',
                 'tokenizer': tokenizer,
                 'eeg_type': args.eeg_type,
                 'bands': args.bands,
                 'add_CLS_token': args.add_CLS_token}
    test_dataset = ZuCo_dataset(**test_args)

    print('trainset size: ',len(train_dataset))
    print('devset size: ',len(dev_dataset))
    print('testset size: ',len(test_dataset))
