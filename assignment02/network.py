import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F

import tqdm

class ThreeInputsNet(nn.Module):
    def __init__(self, n_tokens, n_cat_features, concat_number_of_features, hid_size=64):
        super(ThreeInputsNet, self).__init__()
        self.title_emb = nn.Embedding(n_tokens, embedding_dim=hid_size)
        self.title_conv = nn.Conv1d(hid_size, hid_size, kernel_size=3, padding=1)
        
        self.full_emb = nn.Embedding(num_embeddings=n_tokens, embedding_dim=hid_size)
        self.full_conv = nn.Conv1d(hid_size, hid_size, kernel_size=3, padding=1)
        
        self.category_emb = nn.Embedding(n_cat_features, embedding_dim=hid_size)
        # Correct the input size for the category_out layer
        self.category_out = nn.Linear(n_cat_features * hid_size, hid_size)

        self.inter_dense = nn.Linear(in_features=concat_number_of_features, out_features=hid_size*2)
        self.final_dense = nn.Linear(in_features=hid_size*2, out_features=1)

    def forward(self, whole_input):
        input1, input2, input3 = whole_input
        input1 = input1.long()
        input2 = input2.long()
        input3 = input3.long()
        title_beg = self.title_emb(input1).permute((0, 2, 1))
        title = F.relu(self.title_conv(title_beg))
        title = F.max_pool1d(title, kernel_size=title.size(2)).squeeze(2)

        full_beg = self.full_emb(input2).permute((0, 2, 1))
        full = F.relu(self.full_conv(full_beg))
        full = F.max_pool1d(full, kernel_size=full.size(2)).squeeze(2)

        category = self.category_emb(input3)
        # Flatten the category features before passing them to the category_out layer
        category = self.category_out(category.view(category.size(0), -1))

        concatenated = torch.cat([title, full, category], dim=1)

        out = F.relu(self.inter_dense(concatenated))
        out = self.final_dense(out)

        return out