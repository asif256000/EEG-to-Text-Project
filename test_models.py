import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import numpy as np
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig


class BrainTranslator(nn.Module):
    def __init__(self, pretrained_layers, input_dim=840, embedding_dim=1024, encoder_heads=8, encoder_dim_feedforward=2048):
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
    
    

