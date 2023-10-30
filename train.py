import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import matplotlib.pyplot as plt
from glob import glob
import time
import copy
from tqdm import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

from data_utils import ZuCo_dataset
from models import BrainTranslator
from get_config import get_config
from utils import *


def train(dataloaders, device, model, optimizer, scheduler, num_epochs=25, checkpoint_path_best='./save_data/checkpoints/decoding/best/final.pt',
          checkpoint_path_last='./save_data/checkpoints/decoding/last/final.pt'):
    tic = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for input_embeddings, seq_len, input_masks, input_masks_invert, target_ids, target_masks in tqdm(dataloaders[phase]):

                input_embeddings_batch = input_embeddings.to(device).float()
                input_masks_batch = input_masks.to(device)
                input_masks_invert = input_masks_invert.to(device)
                target_ids_batch = target_ids.to(device)
                target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100

                optimizer.zero_grad()

                model_output = model(input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch)
                loss = model_output.loss

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                
                running_loss += loss.item() * input_embeddings_batch.size(0)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), checkpoint_path_best)
                print(f'[*] Saved best model to {checkpoint_path_best}')

        time_elapsed = time.time() - tic
        print(f'\n[*] Epoch {epoch} completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print('Best val loss: {:4f}'.format(best_loss))
        torch.save(model.state_dict(), checkpoint_path_last)
        print(f'\n[*] Saved last model to {checkpoint_path_last}')

    model.load_state_dict(best_model_wts)
    return model
    

if __name__ == '__main__':
    args = get_config('train')

    num_epochs_step1 = args['num_epoch_step1']
    num_epochs_step2 = args['num_epoch_step2']
    step1_lr = args['learning_rate_step1']
    step2_lr = args['learning_rate_step2']

    batch_size = args['batch_size']

    task_name = args['task_name']

    save_path = args['save_path']
    save_name = os.path.basename(save_path)

    skip_step_one = args['skip_step_one']
    load_step_one = args['load_step_one']
    load_step_one_path = args['load_step_one_path']

    use_random_init = args['use_random_init']

    if use_random_init and skip_step_one:
        step2_lr = 5e-4
    
    if skip_step_one:
        save_name = f'{task_name}_skipstep1_b{batch_size}_sone{step1_lr}_stwo{step2_lr}_eone{num_epochs_step1}_etwo{num_epochs_step2}'
    else:
        save_name = f'{task_name}_2steptraining_b{batch_size}_sone{step1_lr}_stwo{step2_lr}_eone{num_epochs_step1}_etwo{num_epochs_step2}'

    if use_random_init:
        save_name = 'randominit_' + save_name

    output_checkpoint_best = os.path.join(save_path, f'checkpoints_{save_name}', 'best', 'final.pt')
    output_checkpoint_last = os.path.join(save_path, f'checkpoints_{save_name}', 'last', 'final.pt')
    init_dirs([os.path.dirname(output_checkpoint_best), os.path.dirname(output_checkpoint_last)])

    eeg_type = args['eeg_type']
    bands = args['bands']
    print('\n[*] EEG type: ', eeg_type)
    print('[*] Bands: ', bands)

    ''' set random seed '''
    seed_val = 323
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    ''' set device '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('\n[*] Using device: ', device)

    ''' load dataset '''
    input_dataset_list = []
    if 'task1' in task_name:
        print('\n[*] Loading Task1 dataset ...')
        dataset_path_task1 = './dataset/processed/Task1_SR_processed.pickle'
        with open(dataset_path_task1, 'rb') as f:
            input_dataset_list.append(pickle.load(f))
    if 'task2' in task_name:
        print('\n[*] Loading Task2 dataset ...')
        dataset_path_task2 = './dataset/processed/Task2_NR_processed.pickle'
        with open(dataset_path_task2, 'rb') as f:
            input_dataset_list.append(pickle.load(f))
    if 'task3' in task_name:
        print('\n[*] Loading Task3 dataset ...')
        dataset_path_task3 = './dataset/processed/Task3_TSR_processed.pickle'
        with open(dataset_path_task3, 'rb') as f:
            input_dataset_list.append(pickle.load(f))

    ''' save config '''
    print('\n[*] Saving config ...')
    config_save_path = f'./save_data/config/config_{save_name}.pickle'
    init_dirs(os.path.dirname(config_save_path))
    with open(config_save_path, 'wb') as f:
        pickle.dump(args, f)

    ''' load tokenizer '''
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    train_dataset = ZuCo_dataset(input_dataset_list, tokenizer, 'train', eeg_type=eeg_type, bands=bands)        
    dev_dataset = ZuCo_dataset(input_dataset_list, tokenizer, 'dev', eeg_type=eeg_type, bands=bands)

    print('\ntrainset size: ', len(train_dataset))
    print('devset size: ', len(dev_dataset))
    print('\n\n')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    dataloaders = {'train': train_dataloader, 'val': dev_dataloader}

    ''' load model '''
    if use_random_init:
        config = BartConfig.from_pretrained('facebook/bart-large')
        pretrained = BartForConditionalGeneration(config)
    else:
        pretrained = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    model = BrainTranslator(pretrained, input_dim=105*len(bands), embedding_dim=1024, encoder_heads=8, encoder_dim_feedforward=2048)
    model.to(device)

    ''' step one trainig: freeze most of BART params '''
    for name, param in model.named_parameters():
        if 'pretrained' in name and param.requires_grad:
            if ('shared' in name) or ('embed_positions' in name) or ('encoder.layers.0' in name):
                continue
            else:
                param.requires_grad = False

    if skip_step_one:
        if load_step_one:
            model.load_state_dict(torch.load(load_step_one_path))
            print(f'\n[*] Loaded step one model from {load_step_one_path}')
        else:
            print('\n[*] Skip step one training')
    else:
        ''' set optimizer '''
        optimizer_step1 = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=step1_lr, momentum=0.9)
        exp_lr_scheduler_step1 = optim.lr_scheduler.StepLR(optimizer_step1, step_size=20, gamma=0.1)

        ''' set criterion '''
        criterion = nn.CrossEntropyLoss()

        ''' train '''
        print('\n\n=== start Step1 training ... ===')
        model = train(dataloaders, device, model, optimizer_step1, exp_lr_scheduler_step1, 
                        num_epochs=num_epochs_step1, checkpoint_path_best=output_checkpoint_best, checkpoint_path_last=output_checkpoint_last)
    
    ''' step two training: unfreeze all BART params '''
    for name, param in model.named_parameters():
        param.requires_grad = True

    ''' set optimizer '''
    optimizer_step2 = optim.SGD(model.parameters(), lr=step2_lr, momentum=0.9)
    exp_lr_scheduler_step2 = optim.lr_scheduler.StepLR(optimizer_step2, step_size=20, gamma=0.1)

    ''' set criterion '''
    criterion = nn.CrossEntropyLoss()

    ''' train '''
    print('\n\n=== start Step2 training ... ===')
    trained_model = train(dataloaders, device, model, optimizer_step2, exp_lr_scheduler_step2, 
                            num_epochs=num_epochs_step2, checkpoint_path_best=output_checkpoint_best, checkpoint_path_last=output_checkpoint_last)
    