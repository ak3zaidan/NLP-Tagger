from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from mydata.ParityData import *
from models.ParityLSTM import *
from torch.optim import Adam
import torch.nn as nn
import torch
import random
import string
import wandb
wandb.login(key="your key")

# Imports for plotting our result curves
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

use_cuda_if_avail = True
if use_cuda_if_avail and torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

config = {
    "bs":44,   # batch size
    "lr":0.001, # learning rate
    "l2reg":0.0, # weight decay
    "max_epoch":300,
    "linear_warmup":10,
    "train_length":10,
    "eval_length":2500,
    "hidden_dim":16
}

# Main Driver Loop
def main():
    print("Device is: " + device)
    # Build the model and put it on the GPU
    model = ParityLSTM(hidden_dim=config["hidden_dim"])
    model.to(device) # move to GPU if cuda is enabled

    train_loader = getParityDataloader(training=True, max_length=config["train_length"], batch_size = config["bs"])
   
    train(model, train_loader)

    runParityExperiment(model)

    wandb.finish()

# This function evaluate a model on binary strings ranging from length 1 to 20. 
# A plot is saved in the local directory showing accuracy as a function of this length
def runParityExperiment(model):
    lengths = []
    accuracy  = []

    for k in range(config["train_length"], config["eval_length"], 100):
        val_loader = getParityDataloader(training=False, max_length=k, batch_size = config["bs"])
        _, val_acc = evaluate(model, val_loader)
        lengths.append(k)
        accuracy.append(val_acc)
        wandb.log({"Test/acc": val_acc, "Test/length":k})

    f = plt.Figure(figsize=(10,3))
    f.gca().plot(lengths, accuracy)
    f.gca().axvline(x=config["train_length"], c="k", linestyle="dashed")
    f.gca().set_xlabel("Binary String Length")
    f.gca().set_ylabel("Accuracy")
    f.gca().set_xlim(config["train_length"],config["eval_length"])
    f.gca().set_ylim(0.45,1.05)
    f.tight_layout()
    
    wandb.log({"Viz/length":wandb.Image(f)})
    
def train(model, train_loader):
  # Log our exact model architecture string
  config["arch"] = str(model)
  run_name = generateRunName()

  # Startup wandb logging
  wandb.login()
  wandb.init(project="[AI539] Parirty HW2", name=run_name, config=config)

  # Move model to the GPU
  model.to(device)

  # Set up optimizer and our learning rate schedulers
  optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=config["l2reg"])
  linear = LinearLR(optimizer, start_factor=0.25, total_iters=config["linear_warmup"])
  cosine = CosineAnnealingLR(optimizer, T_max = config["max_epoch"]-config["linear_warmup"])
  scheduler = SequentialLR(optimizer, schedulers=[linear, cosine], milestones=[config["linear_warmup"]])

  # Loss
  crit = torch.nn.CrossEntropyLoss()

  # Main training loop with progress bar
  iteration = 0
  for epoch in range(config["max_epoch"]):
    model.train()

    # Log LR
    wandb.log({"LR/lr": scheduler.get_last_lr()[0]}, step=iteration)

    for x, y, lens in train_loader:
      x = x.to(device)
      y = y.to(device)
      lens = lens.to(device)

      out = model(x, lens)
      
      loss = crit(out, y)

      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), 1)
      optimizer.step()
      optimizer.zero_grad()

      acc = (torch.argmax(out, dim=1) == y).to(dtype=float).mean()


      wandb.log({"Loss/train": loss.item(), "Acc/train": acc}, step=iteration)
      iteration+=1

    # Adjust LR
    scheduler.step()
  
def evaluate(model, test_loader):
  model.eval()

  running_loss = 0
  running_acc = 0
  criterion = torch.nn.CrossEntropyLoss(reduction="sum")

  for x,y, lens in test_loader:

    x = x.to(device)
    y = y.to(device)
    lens = lens.to(device)

    out = model(x, lens)
    loss = criterion(out,y)

    acc = (torch.argmax(out, dim=1) == y).to(dtype=float).sum()

    running_loss += loss.item()
    running_acc += acc.item()

  return running_loss/len(test_loader.dataset), running_acc/len(test_loader.dataset)

def generateRunName():
  random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
  run_name = ""+random_string+"_Parity"
  return run_name

wandb.init(
    project="[AI539] Parity HW2",
    name=generateRunName(),
    config=config
)

if __name__== "__main__":
    main()
