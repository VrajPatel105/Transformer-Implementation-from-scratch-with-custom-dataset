import torch
import torch.nn as nn
import math

# Implementation pipeline: 
# Embedding 
# Positional Encoding 
# Multi-Head Attention 
# Add & Norm
# Feedforward
# Add & Norm 
# stack into Encoder layer
#  Decoder (with cross-attention)
#  final Linear + Softmax.





# Embeddings class
class Embedding(nn.Module):

    def __init__(self, x):
        super().__init__()


    def forward(self):

        pass




# Positional Encoding class
class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        positional_encoded_tensor = torch.zeros((max_seq_len, d_model))

        pos = torch.arange(0, max_seq_len).unsqueeze(1).float()
        i = torch.arange(0, d_model, 2).float()
        den = 10000 ** (2*i / d_model)
        final_num_den = pos / den

        positional_encoded_tensor[:, 0::2] = torch.sin(final_num_den)
        positional_encoded_tensor[:, 1::2] = torch.cos(final_num_den)

        self.register_buffer('pe', positional_encoded_tensor.unsqueeze(0))

    def forward(self,x):
        seq_len = x.shape[1]
        x = x + self.pe[:, :seq_len, :]
        return x


# Multi Head attention class
class MultiHeadAttention(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self):

        pass




# Feed forward class
class FeedForward(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self):

        pass




# add and norm class
class AddNorm(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self):

        pass





# residual class
class Residuals(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self):

        pass

