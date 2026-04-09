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

    def __init__(self, batch_size, seq_len, d_model, head_dim):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.head_dim = head_dim
        if (d_model % head_dim != 0):
            raise ValueError("The head dimensions does not fit with d_model")
        self.d_k = d_model // head_dim
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    @staticmethod
    def attention(q,k,v,d_k,mask):
        
        attention_scores = ((q @ k.transpose(-2,-1) ) / math.sqrt(d_k))

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        attention_scores = torch.softmax(attention_scores, dim=-1)

        return attention_scores @ v  


    def forward(self, x, q, k, v):
        self.q = self.W_q(q) # -> q @ W_q
        self.k = self.W_k(k) # -> k @ W_k
        self.v = self.W_v(v) # -> v @ W_v
        # so till now the shape is batch_size, seq_len, d_model for all q,k,v
        # now we need to convert to another tensor shape which is :
        # batch_size,seq_len,d_model => batch_size,seq_len, head_dim, d_k -> batch_size, head_dim, seq_len,d_k  
        self.q = self.q.view(self.batch_size, self.head_dim, self.seq_len, self.d_k).transpose(1,2)
        self.k = self.k.view(self.batch_size, self.head_dim, self.seq_len, self.d_k).transpose(1,2)
        self.v = self.v.view(self.batch_size, self.head_dim, self.seq_len, self.d_k).transpose(1,2)

        self.attention_scores = self.attention(self.q, self.k, self.v, self.d_k, mask=False)
        x = self.W_o(self.attention_scores.transpose(1, 2).contiguous().view(self.batch_size, self.seq_len, self.d_model))
        # summate all here and return 

        return x 


# Feed forward class
class FeedForward(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_model * 4 ,d_model)

    def forward(self,x):
        return self.linear2(self.relu(self.linear1(x)))



# LayerNorm class
class LayerNorm(nn.Module):

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

