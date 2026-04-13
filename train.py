import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split

from tokenizer import Tokenizer

from config import configurations


path = configurations['path']

def load_data(path):
  custom_cols = ["col1", "english_sentence", "col2", "german_sentence"]

  data = pd.read_csv(path, sep='\t', header=None, names=custom_cols, encoding='utf-8')
  data.drop(columns=['col1', 'col2'], inplace=True)
  return data['english_sentence'].tolist(), data['german_sentence'].tolist()




class TranslationDataset(Dataset):

  def __init__(self, eng_sentences, german_sentences, eng_tok, de_tok, max_len):
    super().__init__()
    self.eng_sentences = eng_sentences
    self.german_sentences = german_sentences
    self.eng_tok = eng_tok
    self.de_tok = de_tok
    self.max_len = max_len

  def __len__(self):
    return len(self.eng_sentences)
  
  def __getitem__(self, idx):

    # encoder_input = English, SOS + EOS
    # decoder_input = German, SOS only (no EOS)
    # label = German, EOS only (no SOS)

    encoder_input = self.eng_tok.encode_sentence(self.eng_sentences[idx], add_sos=True, add_eos=True)
    decoder_input = self.de_tok.encode_sentence(self.german_sentences[idx], add_sos=True, add_eos=False)
    encoded_label = self.de_tok.encode_sentence(self.german_sentences[idx], add_sos=False, add_eos=True) # this is for teacher forcing. here since there's no sos, the output will be shifted with one space.

    # Before padding, make sure each encoded list is ≤ max_len. If it's longer, padding would produce negative counts and silently break things.
    assert len(encoder_input) <= self.max_len, f"eng sentence too long: {len(encoder_input)}"
    assert len(decoder_input) <= self.max_len, f"de sentence too long: {len(decoder_input)}"

    # finding the number of padding tokens to be added.
    enc_pad_count = self.max_len - len(encoder_input)
    dec_pad_count = self.max_len - len(decoder_input)
    lbl_pad_count = self.max_len - len(encoded_label)

    # adding the padding token
    encoder_input = encoder_input + [self.eng_tok.PAD_ID] * enc_pad_count
    decoder_input = decoder_input + [self.de_tok.PAD_ID] * dec_pad_count
    encoded_label = encoded_label + [self.de_tok.PAD_ID] * lbl_pad_count

    return {
    "encoder_input": torch.tensor(encoder_input, dtype=torch.long),
    "decoder_input": torch.tensor(decoder_input, dtype=torch.long),
    "label":         torch.tensor(encoded_label, dtype=torch.long),
    }
  

# TESTING workflow till now.
# dataset = TranslationDataset(eng, de, eng_tok, de_tok, max_len=30)
# print(len(dataset))
# sample = dataset[0]
# print(sample["encoder_input"])
# print(sample["decoder_input"])
# print(sample["label"])
# print(sample["encoder_input"].shape)
# ------------------------------------------------------------------------------
# OUTPUT BELOW
# ------------------------------------------------------------------------------
# 3200
# tensor([2, 4, 5, 6, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# tensor([2, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# tensor([4, 5, 6, 7, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# torch.Size([30])

eng, de = load_data(path)
eng_tok = Tokenizer()
eng_tok.build_vocab(eng)

de_tok = Tokenizer()
de_tok.build_vocab(de)

max_len = configurations['max_len'] # 68 -> print(max(len(s.split()) for s in eng))

full_dataset = TranslationDataset(eng, de, eng_tok, de_tok, max_len)
val_size = int(0.1 * len(full_dataset))
train_size = len(full_dataset) - val_size


train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_dataset, batch_size=configurations['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=configurations['batch_size'], shuffle=False)

print(len(train_dataset), len(val_dataset))
batch = next(iter(train_loader))
print(batch["encoder_input"].shape, batch["decoder_input"].shape, batch["label"].shape)