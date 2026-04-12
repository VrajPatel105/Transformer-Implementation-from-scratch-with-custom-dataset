import torch
import torch.nn as nn
import pandas as pd

from config import configurations



path = configurations['path']

custom_cols = ["col1", "english_sentence", "col2", "hindi_sentence"]
def load_data(path):
  data = pd.read_csv(path, sep='\t', header=None, names=custom_cols, nrows=640, encoding='utf-8')
  data.drop(columns=['col1', 'col2'], inplace=True)
  return data['english_sentence'].tolist(), data['hindi_sentence'].tolist()

