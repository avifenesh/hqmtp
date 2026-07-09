"""Loader for engine hidden dumps (BW24_REPLAY_HDUMP) — f32 (legacy) or bf16."""
import json

import numpy as np


class Hiddens:
    """Indexable [n, d] view over an hdump file; __getitem__ returns float32 ndarray."""

    def __init__(self, base):
        self.meta = json.load(open(base + ".hiddens.meta.json"))
        n, d = self.meta["n_tokens"], self.meta["n_embd"]
        self.n, self.d = n, d
        self.dtype = self.meta.get("dtype", "f32")
        if self.dtype == "bf16":
            self.raw = np.memmap(base + ".hiddens", dtype=np.uint16, mode="r",
                                 shape=(n, d))
        else:
            self.raw = np.memmap(base + ".hiddens", dtype=np.float32, mode="r",
                                 shape=(n, d))

    def __getitem__(self, idx):
        rows = np.asarray(self.raw[idx])
        if self.dtype == "bf16":
            return (rows.astype(np.uint32) << 16).view(np.float32)
        return rows
