#!/usr/bin/env python3
"""Patch retrained mtp.* tensors into a copy of the st-ct checkpoint.

Usage:
  python3 patch_head.py <src_model_dir> <retrained_head.safetensors> <dst_model_dir>

The retrained file must contain ONLY mtp.* tensors (same names/shapes/dtypes as the
source checkpoint's). Everything else is hard-linked from the source, and the shard
containing mtp.* tensors is rewritten with the replacements. The result is a normal
HF dir the bw24 SafetensorsSource loads unchanged — the eval sees only the head diff.
"""
import json
import os
import shutil
import sys

from safetensors import safe_open
from safetensors.torch import save_file
import torch


def main() -> None:
    src, head_file, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(dst, exist_ok=True)

    with safe_open(head_file, framework="pt") as f:
        new_head = {n: f.get_tensor(n) for n in f.keys()}
    assert new_head and all(n.startswith("mtp.") for n in new_head), \
        "retrained file must contain only mtp.* tensors"

    index_path = os.path.join(src, "model.safetensors.index.json")
    idx = json.load(open(index_path))
    wmap = idx["weight_map"]

    for name in new_head:
        assert name in wmap, f"{name} not in source checkpoint"
    touched_shards = {wmap[n] for n in new_head}

    for entry in os.listdir(src):
        s, d = os.path.join(src, entry), os.path.join(dst, entry)
        if entry in touched_shards or os.path.isdir(s):
            continue
        if os.path.exists(d):
            os.remove(d)
        os.link(s, d)

    for shard in touched_shards:
        tensors = {}
        with safe_open(os.path.join(src, shard), framework="pt") as f:
            for n in f.keys():
                if n in new_head:
                    old = f.get_tensor(n)
                    new = new_head[n]
                    assert old.shape == new.shape and old.dtype == new.dtype, \
                        f"{n}: shape/dtype mismatch {old.shape}/{old.dtype} vs {new.shape}/{new.dtype}"
                    tensors[n] = new
                else:
                    tensors[n] = f.get_tensor(n)
        save_file(tensors, os.path.join(dst, shard), metadata={"format": "pt"})
        print(f"rewrote {shard} with {sum(1 for n in new_head if wmap[n] == shard)} mtp tensors")

    replaced = sorted(new_head)
    print(f"patched {len(replaced)} tensors -> {dst}")
    for n in replaced:
        print(f"  {n}")


if __name__ == "__main__":
    main()
