import math
import torch
import torch.nn as nn


# initializing the variables
d_model = 4
seq_len = 4 # for current batch 1
batch_size = 2

# W_q = nn.Linear(d_model, d_model)

# q = torch.rand(batch_size, seq_len, d_model)

# print(q)
# q.shape

# print(W_q)


# output = W_q(q)

# print(output)
# output.shape



# print("linear data")

# x = W_q.weight.data
# print(x)
# print(x.shape)
# y = W_q.bias.data
# print(y)
# print(y.shape)



q = torch.tensor([
    # Yellow slice
    [[ 0.62,  0.08, -0.91,  0.84],
     [-0.06,  0.11,  0.51,  1.48],
     [-1.82,  0.93,  1.50,  0.03],
     [-2.35,  1.26,  1.73, -0.73]],
    # Blue slice
    [[-0.54,  0.31,  0.77,  1.33],
     [-0.29,  0.14,  0.62,  1.03],
     [-0.89,  0.34,  1.14,  0.50],
     [-1.40,  0.74,  1.00, -0.57]]
], dtype=torch.float32)  # shape (2, 4, 4)

k = torch.tensor([
    # Yellow slice
    [[ 0.72,  0.45, -0.02,  0.56],
     [ 0.55, -0.31,  1.31,  1.80],
     [ 0.70, -1.96,  1.26,  0.58],
     [-0.48, -1.89,  1.82,  0.88]],
    # Blue slice
    [[ 1.40, -1.11,  0.94,  1.07],
     [ 0.74, -0.63,  0.85,  1.03],
     [ 0.23, -1.00,  1.16,  0.98],
     [-0.32, -1.11,  0.97,  0.36]]
], dtype=torch.float32)  # shape (2, 4, 4)

v = torch.tensor([
    # Yellow slice
    [[ 0.59, -0.64,  1.01,  0.91],
     [-0.47,  0.92,  0.41,  0.73],
     [-1.28,  1.14,  0.63,  0.24],
     [-1.17,  2.16, -0.88,  0.70]],
    # Blue slice
    [[-0.89,  0.36,  1.51,  0.28],
     [-0.62,  0.53,  0.72,  0.24],
     [-0.91,  1.16,  0.01,  0.14],
     [-0.66,  1.21, -0.55,  0.35]]
], dtype=torch.float32)  # shape (2, 4, 4)

# print(f"q before: {q} \n")
# print(f"q before shape  : {q.shape}\n")
# print(f"k before: {k}\n")
# print(f"k before shape: {k.shape}\n")
# print(f"k before : {k} \n")
# print(f"v before shape: {v.shape}\n")



q_len = q.size(1)
k_len = k.size(1)
num_heads = 2
d_k = d_model // num_heads

q_before_transpose = q.view(batch_size, q_len, num_heads, d_k)
k_before_transpose = k.view(batch_size, k_len, num_heads, d_k)
v_before_transpose = v.view(batch_size, k_len, num_heads, d_k)

# print(f"q before tranpose : {q_before_transpose} \n")
# print(f"q before tranpose shape : {q_before_transpose.shape} \n")


q = q.view(batch_size, q_len, num_heads, d_k).transpose(1, 2)
k = k.view(batch_size, k_len, num_heads, d_k).transpose(1, 2)
v = v.view(batch_size, k_len, num_heads, d_k).transpose(1, 2)

# print(f"q : {q} \n")

# print(f"q shape  : {q.shape}\n")
# print(f"k : {k}\n")
# print(f"v : {v}\n")

k_transpose = k.transpose(-2,-1)

# print(f"\nK transpose : {k_transpose}")
# print(f"\nK transpose shape {k_transpose.shape}")

numerator = (q @  k_transpose)
# print(f'\nNumerator : {numerator}')

denominator = math.sqrt(d_k)
# print(f"\ndenominator: {denominator}")

attention_scores = (numerator / denominator)

# print(f"\n atteention_score before softmax ; {attention_scores}")
# print(f" \n attentionscore shape before softmax : {attention_scores.shape}")
attention_scores = torch.softmax(attention_scores, dim=-1)
# print(f"attentino scores after softmax : {attention_scores}")
# print(f"attentino scores shape after softmax : {attention_scores.shape}")

o = attention_scores @ v  
# print(f"\n output o : {o}")
# print(f"\n o shape : {o.shape}")

o.transpose(1,2)
# print("attention scores transpoe : ")
print(attention_scores.transpose(1,2).shape)
print("\n")


# print(o.transpose(1,2).contiguous().shape)

x = o.transpose(1,2).contiguous().view(batch_size, q_len,d_model)

print(x.shape)

# print("applying tha W_o layer")/

W_o = nn.Linear(d_model, d_model, bias=False)

with torch.no_grad():
    W_o.weight.copy_(torch.tensor([
        [ 0.35, -0.52,  0.57,  0.87],
        [ 0.15, -0.68,  0.79,  0.01],
        [ 0.63,  0.97,  0.12,  0.08],
        [ 0.63, -0.04,  0.41,  0.38]
    ], dtype=torch.float32))


x = W_o(o.transpose(1,2).contiguous().view(batch_size, q_len,d_model))

# print(f"after w_0 {x}")

# print(f"after w_o shape {x.shape}")



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
      print(f"\n add x and initial tensor : {x}")
      return self.norm(x)

# initial Tensor : 
initial_tensor = torch.tensor([
    [
        [-0.18,  1.57,  0.49,  0.67],
        [ 1.28,  0.46,  0.73,  1.11],
        [ 0.38, -0.11, -0.64,  1.95],
        [ 0.43, -1.60,  0.18,  1.83]
    ],
    [
        [ 0.81,  1.22, -0.39,  1.64],
        [ 0.84,  0.54,  0.01,  1.00],
        [ 0.91, -0.42,  0.02,  1.00],
        [ 0.14, -0.99,  0.03,  1.00]
    ]
], dtype=torch.float)


# add and norm part
output = initial_tensor + x

print(output)

a = ResidualConnections(d_model)
answer = a(initial_tensor, x)

print(f"norm answer: {answer}")
print(answer.shape)