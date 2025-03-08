import os, sys, ipdb, platform, shutil, zipfile, io, requests, subprocess
from tqdm.notebook import  tqdm
from datetime import datetime
import torch
import torch.nn as nn
from torch.nn import functional as F
import sentencepiece as spm
from dotenv import load_dotenv
from src.config import *
load_dotenv()

class Block(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        head_size= embed_size//n_heads
        self.ma= Multihead(n_heads, head_size)
        self.feed_forward= ForwardLayer(embed_size) 
        self.ln1= nn.LayerNorm(embed_size)
        self.ln2= nn.LayerNorm(embed_size)
    
    def forward(self, x):
        x= x+ self.ma(self.ln1(x))
        x= x+ self.feed_forward(self.ln2(x))
        return x


class ForwardLayer(nn.Module):
    def __init__(self, embed_size, BIAS= True):
        super().__init__()
        self.network= nn.Sequential(
            nn.Linear(embed_size, 6*embed_size, bias=BIAS),
            nn.GELU(),
            nn.Linear(6*embed_size, embed_size, bias=BIAS),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x= self.network(x)
        return x

class Multihead(nn.Module):
    def __init__(self, n_heads, head_size, BIAS=True):
        super().__init__()
        self.heads= nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.combine= nn.Linear(head_size*n_heads, embed_size, bias= BIAS)
        self.dropout= nn.Dropout(dropout)

    def forward(self, x):
        x= torch.cat([head(x) for head in self.heads], dim=-1)
        x= self.combine(x)
        x= self.dropout(x)
        return x

class Head(nn.Module):
    def __init__(self, head_size, BIAS=True):
        super().__init__()
        self.queries= nn.Linear(embed_size, head_size, bias=BIAS)
        self.keys= nn.Linear(embed_size, head_size, bias=BIAS)
        self.values= nn.Linear(embed_size, head_size, bias=BIAS)

        self.register_buffer("tril", torch.tril(torch.ones(context, context)))
        self.dropout= nn.Dropout(dropout)
    
    def forward(self, x): 
        batch_size, sequence_length, vocab_size= x.shape
        q= self.queries(x)
        k= self.keys(x) 
        v= self.values(x) 

        attn_weights= q@k.transpose(-2, -1) * k.shape[-1]**-0.5
        attn_weights= attn_weights.masked_fill(self.tril[:sequence_length, :sequence_length]==0, float("-inf"))
        attn_weights= F.softmax(attn_weights, dim=-1)
        x= attn_weights @ v
        return x
         


