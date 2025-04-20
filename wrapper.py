import train_pos
import random
import numpy as np
import torch

seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

for model_type in ["GRU", "Linear"]:
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        train_pos.config["layers"] = 2 if model_type == "GRU" else 0
        train_pos.main()
