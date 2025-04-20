from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from mydata.PoSData import getUDPOSDataloaders
from models.PoSGRU import PoSGRU
from torch.optim import AdamW
import gensim.downloader
from tqdm import tqdm
import torch.nn as nn
import datetime
import random
import string
import wandb
import torch

use_cuda_if_avail = True
if use_cuda_if_avail and torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

config = {
    "bs": 256,   # batch size
    "lr": 0.0005, # learning rate
    "l2reg": 0.0000001, # weight decay
    "max_epoch": 30,
    "layers": 2,
    "embed_dim": 100,
    "hidden_dim": 256,
    "residual": True,
    "use_glove": True
}

def main():
  # Get dataloaders
  train_loader, val_loader, _, vocab = getUDPOSDataloaders(config["bs"])

  vocab_size = vocab.lenWords()
  label_size = vocab.lenLabels()
  
  ##################################
  #  Q11
  ##################################

  # Preload GloVE vectors
  if config['use_glove']:
    config['embed_dim'] = 100
    embed_init = torch.randn(vocab_size, config['embed_dim'])
    glove_wv = gensim.downloader.load('glove-wiki-gigaword-100')
    count = 0
    missing_words = []

    total_missing = 0
    
    for i, w in vocab.idx2word.items():
        if w in glove_wv:
            embed_init[i,:] = torch.from_numpy(glove_wv[w])
            count += 1
        else:
            total_missing += 1
            if len(missing_words) < 20:
                missing_words.append(w)
    
    print(f'total missing words are {total_missing}')
    print("word vectors loaded: {} / {}".format(count, vocab_size))
    print(f"Missing words examples: {missing_words[:5]}")
  else:
    embed_init = None

  # Build model
  model = PoSGRU(vocab_size=vocab_size, 
                 embed_dim=config["embed_dim"], 
                 hidden_dim=config["hidden_dim"], 
                 num_layers=config["layers"],
                 output_dim=label_size,
                 residual=config["residual"],
                 embed_init=embed_init)
  print(model)

  # Start model training
  train(model, train_loader, val_loader)

def train(model, train_loader, val_loader):
  # Log our exact model architecture string
  config["arch"] = str(model)
  run_name = generateRunName()

  # Startup wandb logging
  wandb.login(key="your key")
  wandb.init(project="[AI539] UDPOS HW2", name=run_name, config=config)

  # Move model to the GPU
  model.to(device)

  ##################################
  #  Q6
  ##################################
  # Set up optimizer and our learning rate schedulers
  optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["l2reg"])
  
  total_steps = config["max_epoch"] * len(train_loader)
  warmup_steps = int(0.1 * total_steps)
  
  warmup_scheduler = LinearLR(optimizer, start_factor=0.25, end_factor=1.0, total_iters=warmup_steps)
  cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps-warmup_steps)
  
  scheduler = SequentialLR(
      optimizer,
      schedulers=[warmup_scheduler, cosine_scheduler],
      milestones=[warmup_steps]
  )

  ##################################
  #  Q7 
  ##################################
  criterion = nn.CrossEntropyLoss(ignore_index=-1)

  # Main training loop with progress bar
  best_val_acc = float('-inf')

  iteration = 0

  pbar = tqdm(total=config["max_epoch"]*len(train_loader), desc="Training Iterations", unit="batch")

  for epoch in range(config["max_epoch"]):
    model.train()

    for x, y, lens in train_loader:
      x = x.to(device)
      y = y.to(device)

      out = model(x)
      
      ##################################
      #  Q7
      ##################################
      batch_size, seq_len, num_labels = out.shape
      loss = criterion(out.reshape(-1, num_labels), y.reshape(-1))

      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      
      nonpad = (y != -1).to(dtype=float).sum().item()
      acc = (torch.argmax(out, dim=2)==y).to(dtype=float).sum() / nonpad

      wandb.log({"Loss/train": loss.item(), "Acc/train": acc.item()}, step=iteration)

      pbar.update(1)
      iteration+=1

    val_loss, val_acc = evaluate(model, val_loader, criterion)

    wandb.log({"Loss/val": val_loss, "Acc/val": val_acc}, step=iteration)
    wandb.log({"LR/lr": scheduler.get_last_lr()[0]}, step=iteration)

    ##################################
    #  Q8
    ##################################
    if val_acc > best_val_acc:
      best_val_acc = val_acc

      torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'best_val_acc': best_val_acc,
        'epoch': epoch
      }, f'best_model_{run_name}.pt')
      
      import pickle
      with open(f'vocab_{run_name}.pkl', 'wb') as f:
        pickle.dump(train_loader.dataset.vocab, f)
      
      print(f"new best modl saved with val accuracy: {best_val_acc:.4f}")

    # Adjust LR
    scheduler.step()

  wandb.finish()
  pbar.close()

def evaluate(model, loader, criterion):
  model.eval()
  total_loss = 0
  correct = 0
  total_tokens = 0
  
  with torch.no_grad():
    for x, y, lens in loader:
      x = x.to(device)
      y = y.to(device)
      
      out = model(x)      
      batch_size, seq_len, num_labels = out.shape
      loss = criterion(out.reshape(-1, num_labels), y.reshape(-1))
      total_loss += loss.item() * batch_size
      
      mask = (y != -1)
      correct += (torch.argmax(out, dim=2) == y).masked_select(mask).sum().item()
      total_tokens += mask.sum().item()
  
  avg_loss = total_loss / len(loader.dataset)
  accuracy = correct / total_tokens if total_tokens > 0 else 0
  
  return avg_loss, accuracy

def generateRunName():
  random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
  now = datetime.datetime.now()
  run_name = ""+random_string+"_UDPOS"
  return run_name

main()
