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

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, self.d_model)
        
    def forward(self, x):

        return self.embedding(x) * math.sqrt(self.d_model) 


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

    def __init__(self, batch_size, seq_len, d_model, num_heads):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    @staticmethod
    def attention(q,k,v,d_k,mask):
        
        attention_scores = ((q @ k.transpose(-2,-1) ) / math.sqrt(d_k))

        if mask is not None:
            mask_bool = (mask == 0)
            attention_scores.masked_fill_(mask_bool == 0, -1e9)
        
        attention_scores = torch.softmax(attention_scores, dim=-1)

        return attention_scores @ v  


    def forward(self, x, q, k, v, mask):
        self.q = self.W_q(q) # -> q @ W_q
        self.k = self.W_k(k) # -> k @ W_k
        self.v = self.W_v(v) # -> v @ W_v

        self.batch_size, self.seq_len, _ = q.shape  # Extract from input!
        head_dim = self.d_model // self.num_heads
        self.d_k = head_dim

        # so till now the shape is batch_size, seq_len, d_model for all q,k,v
        # now we need to convert to another tensor shape which is :
        # batch_size,seq_len,d_model => batch_size,seq_len, head_dim, d_k -> batch_size, head_dim, seq_len,d_k  
        self.q = self.q.view(self.batch_size, self.num_heads, self.seq_len, self.d_k).transpose(1,2)
        self.k = self.k.view(self.batch_size, self.num_heads, self.seq_len, self.d_k).transpose(1,2)
        self.v = self.v.view(self.batch_size, self.num_heads, self.seq_len, self.d_k).transpose(1,2)

        self.attention_scores = self.attention(self.q, self.k, self.v, self.d_k, mask=None)
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

    def __init__(self, d_model, eps = 0.00001):
        super().__init__()
        self.eps = eps 
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))


    def forward(self,x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * ((x - mean) / (std + self.eps)) + self.bias


# residual class
class ResidualConnections(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.norm = LayerNorm(d_model)


    def forward(self,x, sublayer):
      x = x + sublayer
      return self.norm(x)

# Encoder class
'''
What we have to do for encoder class:
get the tensor x that is already positional encoded.
take the tensor x and have it pass through the multiheadattention block
then add the output from multiheadattention block and the initial data x which is obtained by the residualconnections block
and then apply layer norm on this.
now we have x tensor
then apply FNN layer  -> x tensor to x_tensor
then again apply add & norm by residual connections and layernorm.
Now we have our final tensor x ready.
'''

class Encoder(nn.Module):
    def __init__(self, multi_head_attention: MultiHeadAttention, feed_forward: FeedForward, d_model):
        super().__init__()
        self.d_model = d_model
        self.multi_head_attention = multi_head_attention
        self.residual_connection = nn.ModuleList([ResidualConnections(self.d_model) for _ in range(2)])
        self.feed_forward = feed_forward

    def forward(self,x, src_mask):
        sub_layer = self.multi_head_attention(x,x,x,src_mask) # x q k v
        x = self.residual_connection[0](x, sub_layer)
        sub_layer = self.feed_forward(x)
        x = self.residual_connection[1](x, sub_layer)

        # we can also write the above implementation as this : 
        #   x = self.residual_connection[0](x, lambda x: self.multi_head_attention(x, x, x, src_mask))
        #   x = self.residual_connection[1](x, self.feed_forward)

        return x


# Decoder Class





# Combined Transformer class