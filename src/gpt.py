import os, sys, ipdb, platform, shutil, zipfile, io, requests, subprocess
from tqdm.notebook import  tqdm
from datetime import datetime
import torch
import torch.nn as nn
from torch.nn import functional as F
import sentencepiece as spm
from dotenv import load_dotenv

from src.config import *

from src.layers import *


class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embeddings= nn.Embedding(vocab_size, embed_size) 
        self.positions= nn.Embedding(context, embed_size) 
        self.blocks= nn.Sequential(*[Block(n_heads) for _ in range(n_layers)]) 
        self.ln= nn.LayerNorm(embed_size) 
        self.final_linear= nn.Linear(embed_size, vocab_size, bias= BIAS) 
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std= 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input, targets= None):
        loss= None
        BS, SL= input.shape 
        emb= self.embeddings(input) 
        pos= self.positions(torch.arange(SL, device=device)) 
        x= emb + pos
        x= self.blocks(x)
        x= self.ln(x)
        logits= self.final_linear(x)

        if targets is not None:
            BS, SL, VS= logits.shape
            logits= logits.view(BS*SL, VS)
            targets= targets.view(BS*SL)
            loss= F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, input, max_=500,):
        for _ in range(max_):
            input= input[:, -context:]
            logits, _ = self(input)
            logits= logits[:, -1, : ]
            probs= F.softmax(logits, dim= -1)
            next_token= torch.multinomial(probs, num_samples= 1)
            input= torch.cat((input, next_token), dim=1)
        return input

