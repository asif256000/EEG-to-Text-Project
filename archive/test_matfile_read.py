import numpy as np
import pandas as pd
import pickle
import argparse

import os
import glob
import re
from tqdm import tqdm
from scipy.io import loadmat
import h5py


def load_mat_files(filepath):
    try:
        mat_data = loadmat(filepath, squeeze_me=True, struct_as_record=False)['sentenceData']
    except:
        mat_data = h5py.File(filepath, 'r')
        mat_data = mat_data['sentenceData']
    return mat_data

def read_mat_file(mat_data):
    for sent in mat_data:
        word_data = sent.word
        if isinstance(word_data, float):
            return None

        data = {'content': sent.content}    # get the content (sentences)
        data['sentence_level_EEG'] = {'mean_t1':sent.mean_t1, 'mean_t2':sent.mean_t2, 'mean_a1':sent.mean_a1, 
                                      'mean_a2':sent.mean_a2, 'mean_b1':sent.mean_b1, 'mean_b2':sent.mean_b2,
                                      'mean_g1':sent.mean_g1, 'mean_g2':sent.mean_g2}       # read sentence-level EEG data
        data['word'] = []

        word_token_has_fixations = []
        word_token_with_mask = []
        word_token_all = []

        for word in word_data:
            word_object = {'content': word.content}
            word_token_all.append(word.content)

            word_object['n_fixations'] = word.nFixations
            if word.nFixations > 0:    
                word_object['word_level_EEG'] = {'FFD':{'FFD_t1':word.FFD_t1, 'FFD_t2':word.FFD_t2, 'FFD_a1':word.FFD_a1, 
                                                        'FFD_a2':word.FFD_a2, 'FFD_b1':word.FFD_b1, 'FFD_b2':word.FFD_b2, 
                                                        'FFD_g1':word.FFD_g1, 'FFD_g2':word.FFD_g2}}
                word_object['word_level_EEG']['TRT'] = {'TRT_t1':word.TRT_t1, 'TRT_t2':word.TRT_t2, 'TRT_a1':word.TRT_a1, 
                                                        'TRT_a2':word.TRT_a2, 'TRT_b1':word.TRT_b1, 'TRT_b2':word.TRT_b2, 
                                                        'TRT_g1':word.TRT_g1, 'TRT_g2':word.TRT_g2}
                word_object['word_level_EEG']['GD'] = {'GD_t1':word.GD_t1, 'GD_t2':word.GD_t2, 'GD_a1':word.GD_a1, 
                                                       'GD_a2':word.GD_a2, 'GD_b1':word.GD_b1, 'GD_b2':word.GD_b2, 
                                                       'GD_g1':word.GD_g1, 'GD_g2':word.GD_g2}
                data['word'].append(word_object)
                word_token_has_fixations.append(word.content)
                word_token_with_mask.append(word.content)
            else:
                # NOTE:if a word has no fixation, simply skip it
                continue

        data['word_token_has_fixations'] = word_token_has_fixations
        data['word_token_with_mask'] = word_token_with_mask
        data['word_token_all'] = word_token_all
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    args = parser.parse_args()

    task_dir = './dataset/raw/{}'.format(args.task)

    matfiles = glob.glob(os.path.join(task_dir, '*.mat'))
    matfiles = sorted(matfiles)

    if len(matfiles) == 0:
        raise ValueError('No mat files found in {}'.format(task_dir))
    
    dataset = {}
    for matfile in tqdm(matfiles):
        subject_name = re.sub('results', '', os.path.basename(matfile).split('_')[0])
        dataset[subject_name] = []

        matdata = load_mat_files(matfile)
        word = read_mat_file(matdata)

        dataset[subject_name].append(word)

    # save file
    output_name = './dataset/processed/{}_processed.pickle'.format(args.task)
    if not os.path.exists(os.path.dirname(output_name)):
        os.makedirs(os.path.dirname(output_name))
    with open(output_name, 'w') as f:
        print(f'\n[*] Saving file to {output_name}')
        pickle.dump(dataset, f)

    # validate dataset save
    with open(output_name, 'r') as f:
        dataset = pickle.load(f)
        print(f'\n[*] Dataset loaded from {output_name}')
        print(f'\n[*] Subjects : {dataset.keys()}')