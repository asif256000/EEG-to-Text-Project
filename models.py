"""
This file contains the brain decoder model. This model is used to decode the EEG signals into words. It consists of 
an additional transformer layer before a pretrained BART model. The brain is assumed to be the encoder and the EEG 
signals as the eeg-word embedding. These eeg-word embeddings are then fed through a into the pretrained BART model to generate
word embeddings. The word embeddings are then compared to the target word embeddings to compute the loss.
"""

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import numpy as np


class BrainTranslator(nn.Module):
    def __init__(self, pretrained_layers, input_dim=840, embedding_dim=1024, encoder_heads=8, encoder_dim_feedforward=2048):
        '''
        :param pretrained_layers: pretrained BART model
        :param input_dim: input dimension of EEG signals
        :param embedding_dim: embedding dimension of EEG signals embedding
        :param encoder_heads: number of heads in the encoder layer
        :param encoder_dim_feedforward: dimension of the feedforward layer in the encoder layer
        '''
        super(BrainTranslator, self).__init__()

        self.pretrained = pretrained_layers
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=encoder_heads, dim_feedforward=encoder_dim_feedforward,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)

        self.fc1 = nn.Linear(input_dim, embedding_dim)

    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted):
        encoded_embeddings = self.encoder(input_embeddings_batch, src_key_padding_mask=input_masks_invert)
        encoded_embeddings = F.relu(self.fc1(encoded_embeddings))
        out = self.pretrained(inputs_embeds=encoded_embeddings, attention_mask=input_masks_batch, return_dict=True,
                              labels=target_ids_batch_converted)
        return out
    
    

