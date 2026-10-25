import random
import numpy as np
import torch
import torch.nn.functional as F
import math

CFG = {
    # MODEL
    "seq_len": 4096,
    "d_model": 512,
    "n_heads": 8,
    "n_layers": 48,
    "d_ff": None,  # None: 8/3 * d_model, rounded to a multiple of 128
    "dropout": 0.0,
    "tie_weights": True,
    "N_min": 64,
    "N_max": 256,
    "k_lm": 32,

    # DATA
    "tokens_per_shard": 500_000_000,
    "target_shards": 30,
    "val_shards": 1,

    # TRAINING
    "seed": 6767,  # for multi-seed runs change this and save_dir together
    "batch_size": 16,
    "grad_accum": 16,
    "max_steps": 5000,
    "warmup_steps": 500,
    "lr": 2e-4,
    "weight_decay": 0.0,
    "grad_clip": 0.5,
    "val_every": 250,
    "log_every": 20,
    "metrics_every": 250,
    "gradient_checkpointing": False,
    "compile": True,
    "loss_type": "liger",
    "save_dir": "./checkpoints_dsalt_48L_6767seed",
    "max_val_batches": 100,
}

def set_seed(seed=999):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
