from src.config import *
import datetime, os
import wandb
import torch
import sentencepiece as spm
from dotenv import load_dotenv
load_dotenv()
wandb_key= os.getenv("wandb_key")


encode= lambda sp, s:sp.Encode(s)
decode= lambda sp, l:sp.Decode(l)

def init_wandb(wandb_log= True, wandb_key= wandb_key):
    if wandb_log:
        wandb.login(key=wandb_key)
        wandb.init(project= wandb_project, name=wandb_run_name)
        return True
    return False

def read_data(wiki_path= wiki_path):
    with open (wiki_path, "r", encoding= "utf-8") as f:
          text= f.read()
    return text

def tokenize(model_file=tokenizer_model_file_path):
    sp= spm.SentencePieceProcessor(model_file= tokenizer_model_file_path)
    vocab_size= sp.GetPieceSize()
    return sp, vocab_size

def save_encoded_data(sp, path= None):
    if os.path.exists(path):
        data= torch.load(path)
    else:
        text= read_data(wiki_path)
        data= torch.tensor(encode(sp=sp, s=text), dtype=torch.long)
        torch.save(data, encoded_data_path)
    return data

def get_batch(data, split:str="train"):
    # BS- batch size | SL- Sequence length
    data_size= len(data)
    spl= int(0.9*data_size)
    train_data= data[:spl]
    val_data= data[spl:]

    data= train_data if split=="train" else val_data
    inds= torch.randint(len(data)- context, (batch_size,))
    x= torch.stack([data[i: i+context] for i in inds])
    y= torch.stack([data[i+1: i+context+1] for i in inds])
    x,y= x.to(device), y.to(device)
    return x,y

def load_checkpoints(path, model, optimizer, scheduler):
    print(f"LLM- Loading model from path {path}")
    checkpoint= torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration= checkpoint['iteration']
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    loss= checkpoint['loss']
    print(f"Checkpoint iter: {iteration} | Loss: {loss}")
    return iteration, loss

def init_optimizer(model):
    p_dict= {p_name:p for p_name, p in model.named_parameters() if p.requires_grad}
    weight_decay_p= [p for n,p in p_dict.items() if p.dim() >=2]
    no_weight_decay_p= [p for n,p in p_dict.items() if p.dim() <2]
    optimizer_groups= [
        {'params': weight_decay_p,
        'weight_decay': weight_decay},
        {'params':no_weight_decay_p, 
        'weight_decay':0.0}
    ]

    optimizer= torch.optim.AdamW(optimizer_groups, lr=lr, betas= (0.9, 0.99))
    return optimizer

def init_scheduler(optimizer):
    scheduler= torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_iters, eta_min=lr/10)
    return scheduler

@torch.no_grad()
def calculate_loss(model):
    out={}
    model.eval()
    for split in ['train', 'eval']:
        l= torch.zeros(eval_iters)
        for i in range(eval_iters):
            x,y= get_batch(split="train")
            _, loss= model(x,y)
            l[i]= loss
        out[split]= l.mean().item()
    model.train()
    return out