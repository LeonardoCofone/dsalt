import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

from utils.utils_dsalt import CFG

SHARDS_DIR = Path("./data_shards")
SEQ_LEN    = CFG["seq_len"]


class LongformShardDataset(Dataset):
    def __init__(self, prefix: str):
        shard_paths = sorted(SHARDS_DIR.glob(f"{prefix}_*.pt"))
        shard_paths = [p for p in shard_paths if not p.name.endswith(".bounds.pt")]
        if not shard_paths:
            raise FileNotFoundError(f"no shards for prefix {prefix!r} in {SHARDS_DIR}")

        self.windows = []
        self._shards = []
        n_docs = n_tok = 0

        for sp in shard_paths:
            bp = sp.with_suffix(".bounds.pt")
            tokens = torch.load(sp, map_location="cpu")
            bounds = torch.load(bp, map_location="cpu").tolist() if bp.exists() else [len(tokens)]
            si = len(self._shards)
            self._shards.append(tokens)
            n_tok += len(tokens)

            prev = 0
            for end in bounds:
                n_docs += 1
                n_win = (end - prev) // (SEQ_LEN + 1)
                for w in range(n_win):
                    self.windows.append((si, prev + w * (SEQ_LEN + 1)))
                prev = end

        print(f"[dataset:{prefix}] {len(shard_paths)} shards | {n_tok:,} tokens | "
              f"{n_docs:,} docs | {len(self.windows):,} sequences")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        si, start = self.windows[i]
        s = self._shards[si][start : start + SEQ_LEN + 1].long()
        return s[:-1].clone(), s[1:].clone()


FalconShardDataset = LongformShardDataset


def dsalt_collate_fn(batch):
    input_ids_list = [torch.as_tensor(item[0]) for item in batch]
    labels_list    = [torch.as_tensor(item[1]) for item in batch]
    seqlens        = [x.shape[0] for x in input_ids_list]
    cu_seqlens     = torch.zeros(len(seqlens) + 1, dtype=torch.int32)
    torch.cumsum(torch.tensor(seqlens, dtype=torch.int32), dim=0, out=cu_seqlens[1:])
    max_seqlen     = max(seqlens)
    return torch.cat(input_ids_list, dim=0), torch.cat(labels_list, dim=0), cu_seqlens, max_seqlen


def make_loaders(limit=None):
    train_ds = LongformShardDataset("train_shard")
    val_ds   = LongformShardDataset("val_shard")

    if limit is not None:
        train_ds.windows = train_ds.windows[:limit]

    train_loader = DataLoader(
        train_ds, batch_size=CFG["batch_size"], shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
        persistent_workers=True, collate_fn=dsalt_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG["batch_size"], shuffle=False,
        num_workers=2, pin_memory=True, drop_last=True,
        persistent_workers=True, collate_fn=dsalt_collate_fn,
    )

    eff = CFG["batch_size"] * CFG["grad_accum"] * CFG["seq_len"]
    print(f"Effective tokens/step: {eff:,}")
    print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")
    return train_loader, val_loader, CFG
