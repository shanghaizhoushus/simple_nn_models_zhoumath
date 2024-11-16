# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 14:09:52 2024

@author: zhoushus
"""

# Import Packages
import os
import torch
from torch import nn
import torch.nn.functional as F
import warnings

#Setting
num_cpus = os.cpu_count()
torch.set_num_threads(num_cpus-1)
warnings.filterwarnings("ignore")

# Define a simple logistic regression model
class SimpleLR(nn.Module):
    
    # Initializes the SimpleLR class
    def __init__(self, input_dim):
        super(SimpleLR, self).__init__()
        self.output = nn.Linear(input_dim, 1)

    # Defines the forward pass of the model
    def forward(self, x, unused_arg=None):
        x = self.output(x)
        return x

# Define a simple neural network model
class SimpleNN(nn.Module):
    
    # Initializes the SimpleNN class
    def __init__(self, input_dim, hidden_sizes):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_sizes[0])
        self.layer2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.output = nn.Linear(hidden_sizes[1], 1)

    # Defines the forward pass of the model
    def forward(self, x, unused_arg=None):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.output(x)
        return x

# Define a general neural network model
class GeneralNN(nn.Module):
    
    # Initializes the SimpleLR class
    def __init__(self, input_dim, hidden_sizes, dropout_rate):
        super(GeneralNN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        
        for h in hidden_sizes:
            self.hidden_layers.append(nn.Linear(input_dim, h))
            self.hidden_layers.append(nn.BatchNorm1d(h))
            input_dim = h

        self.output = nn.Linear(input_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)

    # Defines the forward pass of the model
    def forward(self, x, unused_arg=None):
        
        for i in range(0, len(self.hidden_layers), 2):
            x = self.hidden_layers[i](x)
            x = self.hidden_layers[i+1](x)
            x = F.relu(x)
            x = self.dropout(x)

        x = self.output(x)
        return x

# Define a simple FTTransformer model
class SimpleFTTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_heads=8, num_layers=3,
                 ff_hidden_dim=256, dropout=0.1):
        
        # Initializes the SimpleFTTransformer class
        super(SimpleFTTransformer, self).__init__()
        self.feature_embedding = nn.ModuleList([
            nn.Linear(1, embed_dim) for _ in range(input_dim)
        ])
        self.column_embedding = nn.Parameter(torch.randn(input_dim, embed_dim))
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads, ff_hidden_dim, dropout) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(embed_dim, 1)

    # Defines the forward pass of the model
    def forward(self, x, unused_arg=None):
        batch_size, num_features = x.shape
        x = x.unsqueeze(-1)
        feature_embeds = [self.feature_embedding[i](x[:, i]) for i in range(num_features)]
        feature_embeds = torch.stack(feature_embeds, dim=1)
        H = feature_embeds + self.column_embedding
        
        for layer in self.transformer_layers:
            H = layer(H)
            
        H = H.mean(dim=1)
        output = self.output_layer(H)
        return output

# Define the TransformerLayer
class TransformerLayer(nn.Module):
    
    # Initializes the TransformerLayer class
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout):
        super(TransformerLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads,
                                                    dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    # Defines the forward pass of the model
    def forward(self, x):
        attn_output, _ = self.self_attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
    