import pickle
import numpy as np
from tqdm import tqdm
import os

from transformers import BartTokenizer, BartForConditionalGeneration
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import torch

from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

from get_config import get_config
from data_utils import ZuCo_dataset
from models import BrainTranslator


def eval(dataloader, device, tokenizer, criterion, model, output_results_path='./save_data/eval_results/results.txt'):
    os.makedirs(os.path.dirname(output_results_path), exist_ok=True)

    model.eval()
    running_loss = 0.0

    target_token_list = []
    target_string_list = []
    pred_token_list = []
    pred_string_list = []
    
    with open(output_results_path, 'w') as f:
        for input_embeddings, seq_len, input_masks, input_masks_invert, target_ids, target_masks in tqdm(dataloader['test']):
            # get eeg-word embedding tensor and target word embedding tensor
            input_embeddings_batch = input_embeddings.to(device).float()
            input_masks_batch = input_masks.to(device)
            input_masks_invert = input_masks_invert.to(device)
            target_ids_batch = target_ids.to(device)

            # decode target word embedding tensor to get target string
            target_tokens = tokenizer.convert_ids_to_tokens(target_ids_batch[0].tolist(), skip_special_tokens=True)
            target_string = tokenizer.decode(target_ids_batch[0].tolist(), skip_special_tokens=True)
            f.write('Target string: {}\n'.format(target_string))

            target_token_list.append(target_tokens)
            target_string_list.append(target_string)

            target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100

            # use brain decoder model to get reconstructed string from eeg-word embedding
            model_output = model(input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch)
            loss = model_output.loss

            logits = model_output.logits
            probs = logits[0].softmax(dim=1)
            values, predictions = probs.topk(1)
            predictions = predictions.squeeze()
            predicted_string = tokenizer.decode(predictions).split('</s></s>')[0].replace('<s>', '')
            # save in a txt file
            f.write('Predicted string: {}\n'.format(predicted_string))
            f.write('--------------------------------------------------------------\n\n\n')

            predictions = predictions.tolist()
            truncated_predictions = []
            for pred in predictions:
                if pred != tokenizer.eos_token_id:
                    truncated_predictions.append(pred)
                else:
                    break
            
            pred_tokens = tokenizer.convert_ids_to_tokens(truncated_predictions, skip_special_tokens=True)
            pred_token_list.append(pred_tokens)
            pred_string_list.append(predicted_string)

            running_loss += loss.item() * input_embeddings_batch.size()[0]
        
        running_loss = running_loss / len(dataloader['test'].dataset)
        print('Test Loss: {}'.format(running_loss))

        ''' corpus rouge score '''
        rouge = Rouge()
        rouge_score = rouge.get_scores(pred_string_list, target_string_list, avg=True)
        rouge1_score = rouge_score['rouge-1']

        print('rouge-1 score P: ', rouge1_score['p'])
        print('rouge-1 score R: ', rouge1_score['r'])
        print('rouge-1 score F: ', rouge1_score['f'])

        ''' corpus bleu score '''
        list_of_references = [[tt] for tt in target_token_list]

        bleu1_score = corpus_bleu(list_of_references, pred_token_list, weights=(1, 0, 0, 0))
        bleu2_score = corpus_bleu(list_of_references, pred_token_list, weights=(0, 1, 0, 0))
        bleu3_score = corpus_bleu(list_of_references, pred_token_list, weights=(0, 0, 1, 0))
        bleu4_score = corpus_bleu(list_of_references, pred_token_list, weights=(0, 0, 0, 1))

        print('BLEU1: ', bleu1_score)
        print('BLEU2: ', bleu2_score)
        print('BLEU3: ', bleu3_score)
        print('BLEU4: ', bleu4_score)


if __name__ == '__main__':
    args = get_config('eval')

    batch_size = 1

    ''' load training config'''
    training_config = pickle.load(open(args['config_path'], 'rb'))

    output_results_path = os.path.join('./save_data/eval_results', f"results_{os.path.basename(training_config['save_path'])}")

    eeg_type = training_config['eeg_type']
    bands = training_config['bands']
    print('\n[*] EEG type: ', eeg_type)
    print('[*] Bands: ', bands)

    task_name = training_config['task_name']
    print(f'\n[*] Task name: {task_name}')

    seed_val = 213
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

    ''' load tokenizer '''
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    test_dataset = ZuCo_dataset(input_dataset_list, tokenizer, 'test', eeg_type=eeg_type, bands=bands)
    print('--------------------------------------------------------------')
    print('testset size: ', len(test_dataset))
    print('--------------------------------------------------------------')
    print('\n\n')

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    dataloaders = {'test': test_dataloader}

    ''' load model '''
    model_path = args['checkpoint_path']
    pretrained = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    model = BrainTranslator(pretrained, input_dim=105*len(bands), embedding_dim=1024, encoder_heads=8, encoder_dim_feedforward=2048)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    ''' eval '''
    test_loss, rouge_score, bleu_score = eval(dataloaders, device, tokenizer, criterion, model, output_results_path)

