import os
import torch
from torch.utils.data import DataLoader, DistributedSampler
from utils.data_utils import make_loaders, FalconShardDataset
from utils.utils_dsalt import CFG, set_seed
from dsalt.training.gpu_auto import setup_ddp
from dsalt import DSALTConfig, DSALTLMHeadModel
from dsalt.training import DSALTTrainer

import importlib.metadata
import dsalt

os.environ["TORCHDYNAMO_VERBOSE"] = "1"

VOCAB_SIZE = 50257

def train_fn():
    rank, local_rank, world_size = setup_ddp()
    set_seed(CFG.get("seed", 999) + rank)
    print(f"seed: {CFG.get('seed', 999)}")
    train_loader, val_loader, cfg = make_loaders(limit=None)

    def dsalt_collate_fn(batch):
        input_ids_list = [torch.as_tensor(item[0]) for item in batch]
        labels_list    = [torch.as_tensor(item[1]) for item in batch]
        seqlens        = [x.shape[0] for x in input_ids_list]
        cu_seqlens     = torch.zeros(len(seqlens) + 1, dtype=torch.int32)
        torch.cumsum(torch.tensor(seqlens, dtype=torch.int32), dim=0, out=cu_seqlens[1:])
        max_seqlen     = max(seqlens)
        return torch.cat(input_ids_list, dim=0), torch.cat(labels_list, dim=0), cu_seqlens, max_seqlen

    train_sampler = DistributedSampler(
        train_loader.dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=CFG.get("seed", 999),
    )
    val_sampler = DistributedSampler(
        val_loader.dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )
    train_loader = DataLoader(
        train_loader.dataset,
        batch_size=CFG["batch_size"],
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        collate_fn=dsalt_collate_fn,
    )
    val_loader = DataLoader(
        val_loader.dataset,
        batch_size=CFG["batch_size"],
        sampler=val_sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        collate_fn=dsalt_collate_fn,
    )

    model_cfg = DSALTConfig(
        vocab_size=VOCAB_SIZE,
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        n_min=cfg["N_min"],
        n_max=cfg["N_max"],
        k_lmk=cfg["k_lm"],
        d_ff=cfg["d_ff"],
        max_seq_len=cfg["seq_len"],
        dropout=cfg["dropout"],
        loss_fn=cfg["loss_type"],
    )

    if rank == 0:
        os.makedirs(cfg["save_dir"], exist_ok=True)
        model_cfg.save(os.path.join(cfg["save_dir"], "model_config.json"))

    dsalt_model = DSALTLMHeadModel.from_config(model_cfg)

    trainer = DSALTTrainer(
        model=dsalt_model,
        train_loader=train_loader,
        val_loader=val_loader,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        max_grad_norm=cfg["grad_clip"],
        warmup_steps=cfg["warmup_steps"],
        total_steps=cfg["max_steps"],
        grad_accum=cfg["grad_accum"],
        log_every=cfg["log_every"],
        val_every=cfg["val_every"],
        max_val_batches=cfg["max_val_batches"],
        save_every=cfg["val_every"],
        save_dir=cfg["save_dir"],
        mixed_precision="auto",
        gradient_checkpointing=cfg["gradient_checkpointing"],
        compile_model=cfg["compile"],
        metrics_every=cfg["metrics_every"],
    )

    trainer.train()

if __name__ == "__main__":
    train_fn()
