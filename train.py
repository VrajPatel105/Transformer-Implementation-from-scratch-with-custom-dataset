import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from config import configurations


path = configurations['path']

def load_data(path):
  custom_cols = ["col1", "english_sentence", "col2", "german_sentence"]

  data = pd.read_csv(path, sep='\t', header=None, names=custom_cols, encoding='utf-8')
  data.drop(columns=['col1', 'col2'], inplace=True)
  return data['english_sentence'].tolist(), data['german_sentence'].tolist()

# Subclass torch.utils.data.Dataset
# __init__ stores everything you'll need (sentences, tokenizers, max_len)
# __len__ returns how many pairs
# __getitem__(idx):

# Encode English with SOS + EOS
# Encode German with SOS only → this is decoder_input
# Encode German with EOS only → this is label
# Pad all three to max_len with PAD_ID
# Convert to torch.tensor(..., dtype=torch.long)
# Return as a dict

