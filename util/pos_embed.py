# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

import numpy as np

import torch

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------


def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: tuple of (depth, height, width)
    return:
    pos_embed: [grid_size[0]*grid_size[1]*grid_size[2], embed_dim] or [1+grid_size[0]*grid_size[1]*grid_size[2], embed_dim] (w/ or w/o cls_token)
    """
    grid_d = np.arange(grid_size[0], dtype=np.float32)  # Use grid_size[0] for depth
    grid_h = np.arange(grid_size[1], dtype=np.float32)  # Use grid_size[1] for height
    grid_w = np.arange(grid_size[2], dtype=np.float32)  # Use grid_size[2] for width
    grid = np.meshgrid(grid_w, grid_h, grid_d, indexing='ij')  # Create a 3D grid
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, grid_d.size, grid_h.size, grid_w.size])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


# Modified:now compatible with non even embed_dim
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    half_dim = embed_dim // 2  # Half the embedding dimension
    omega = np.arange(half_dim, dtype=np.float32)
    omega /= half_dim
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    if embed_dim % 2 == 1:
        # If embed_dim is odd, add an extra sin component for the last dimension
        extra_sin = np.sin(pos * (1. / 10000)).reshape(-1, 1)
        emb = np.concatenate([emb_sin, emb_cos, extra_sin], axis=1)  # (M, D)
    else:
        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)

    return emb

# modified for non-divisible embedding compatabiltiy
def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    # Calculate individual dimensions for depth, height, and width
    base_dim = embed_dim // 3
    remainder = embed_dim % 3

    dim_d = base_dim
    dim_h = base_dim + (1 if remainder > 1 else 0)
    dim_w = base_dim + (1 if remainder > 0 else 0)

    # Use calculated dimensions to encode each grid dimension
    emb_d = get_1d_sincos_pos_embed_from_grid(dim_d, grid[0])  # Depth
    emb_h = get_1d_sincos_pos_embed_from_grid(dim_h, grid[1])  # Height
    emb_w = get_1d_sincos_pos_embed_from_grid(dim_w, grid[2])  # Width

    emb = np.concatenate([emb_d, emb_h, emb_w], axis=1)  # Concatenate embeddings
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_h.size, grid_w.size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb




# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed
