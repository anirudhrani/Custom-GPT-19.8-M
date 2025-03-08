import os, torch
from datetime import datetime

# WandB
wandb_project= "llm1"    
wandb_run_name= "llm1-"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# Paths
os.makedirs("data", exist_ok=True)
tokenizer_model_file_path= os.path.join("data", "wiki_tokenizer.model")
wiki_path= os.path.join("data", "wiki.txt")
encoded_data_path= os.path.join("data", "encoded_data.pt")

# Architecture Parameters
batch_size= 32 # 8 to 128
context= 512 # context length
embed_size=  384 # Mathematical vector of 384 no. representing the semantic meaning in dimension
n_layers= 7 # Transformer Blocks
n_heads= 7  # 
BIAS= True

# Hyperparams
lr= 3e-4
dropout= 0.05
weight_decay= 0.01 # L2 regularization.(Keeps weights small to mitigate bias towards one particular feature.)
grad_clip= 1.0 # What direction to tweak the weight to learn.as_integer_ratio

# Training params
train_iters= 100000
eval_interval= 50 # 
eval_iters= 10  # Size of eval data
compile= False # aids in better memory management
checkpoint_dir= "models/" # Intermediate stages of params
checkpoint_fn= "latest.pt" # File name for saving checkpoints
checkpoint_load_fn= "latest.pt" # File name for loading checkpoints
dtype= torch.bfloat16
load_pretrained= True
start_iteration=0
best_val_loss= float('inf')

# Mode
inference= False
device= "mps" if torch.backends.mps.is_available() else "cpu"
