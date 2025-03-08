import os, sys, ipdb, platform, shutil, zipfile, io, requests, subprocess
from tqdm.notebook import  tqdm
from datetime import datetime
import torch
import torch.nn as nn
from torch.nn import functional as F
import sentencepiece as spm
from dotenv import load_dotenv

from src.config import *
from src.utils import *
from src.gpt import GPT
from src.layers import *

device= "mps" if torch.backends.mps.is_available() else "cpu"

init_wandb()

sp, vocab_size= tokenize()
print("Tokenized")
model= GPT(vocab_size=vocab_size)
model= model.to(dtype)
model= model.to(device)

if compile:
    print("Torch :: Compiling Model")
    model= torch.compile(model)

print(sum(p.numel() for p in model.parameters())/ 1e6, "Million parameters")

data= save_encoded_data(sp, encoded_data_path)



# l= calculate_loss()
# print(l)

if os.path.exists(f"{checkpoint_dir}/{checkpoint_load_fn}") and load_pretrained:
    print("Here")
    start_iteration, loss= load_checkpoints(checkpoint_dir+checkpoint_load_fn)
    best_val_loss= loss

optimizer= init_optimizer(model)
scheduler= init_scheduler(optimizer)

# Train Loop
try:
    print("entered Train loop")
    for i in tqdm(range(start_iteration, train_iters), desc="Training Progress"):
        xb, yb= get_batch(data, "train")
        logits, loss= model(xb, yb)
        
        if (i% eval_interval==0 or i== train_iters-1):
            l= calculate_loss(model=model)
            print(f"\n{i} | Train loss: {l['train']} | Val Loss: {l['eval']}")
            print(f"\n{generate_sample('Once upon a time')}")

            if l['eval']< best_val_loss:
                best_val_loss= l['eval']
                print(f"----------- Saving Checkpoints. Best Val loss: {best_val_loss} -----------")
                torch.save({
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'loss':best_val_loss,
                    'scheduler_state_dict': scheduler.state_dict(),
                    'iteration':i,
                }, checkpoint_dir+checkpoint_fn)
            
            if wandb_log:
                wandb.log({
                    "loss/train":l['train'],
                    "loss/val":l['eval'],
                    "lr": scheduler.get_last_lr()[0],
                },
                step= i)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()

    if init_wandb():
        wandb.finish()
        
except KeyboardInterrupt:
    print("Training Interrupted Cleaning up")

except Exception as e:
    print(e)

finally:
    torch.mps.empty_cache()
    print("GPU memory released.")
    sys.exit(0)