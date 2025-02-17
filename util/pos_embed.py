# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

import numpy as np

import torch

import torch.nn.functional as F


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


def interpolate_pos_embed(model, checkpoint_model):
    """Interpolate positional embeddings when loading a pre-trained model with a different input size in 3D ViTs."""
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]  # Hidden dimension
        num_patches = model.patch_embed.num_patches  # New number of patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches  # Extra tokens (CLS, etc.)

        # Compute original and new patch grid sizes
        num_patches_old = pos_embed_checkpoint.shape[1] - num_extra_tokens  # Old number of patches
        num_patches_new = num_patches  # New number of patches

        # Infer original patch grid dimensions
        orig_d = int(round(num_patches_old ** (1 / 3)))
        orig_h = orig_d
        orig_w = orig_d

        # Infer new patch grid dimensions
        new_d = int(round(num_patches_new ** (1 / 3)))
        new_h = new_d
        new_w = new_d

        if orig_d * orig_h * orig_w != num_patches_old or new_d * new_h * new_w != num_patches_new:
            raise ValueError(
                f"Patch embedding count mismatch! Old: {num_patches_old}, New: {num_patches_new}. "
                f"Check your patching setup."
            )

        if (orig_d, orig_h, orig_w) != (new_d, new_h, new_w):
            print(f"Interpolating position embedding from {orig_d}x{orig_h}x{orig_w} to {new_d}x{new_h}x{new_w}")

            # Extract extra tokens (CLS, distillation, etc.)
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]

            # Reshape into (D, H, W, Embedding)
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(1, orig_d, orig_h, orig_w, embedding_size).permute(0, 4, 1, 2, 3)

            # Perform 3D interpolation
            pos_tokens = F.interpolate(pos_tokens, size=(new_d, new_h, new_w), mode='trilinear', align_corners=False)

            # Reshape back
            pos_tokens = pos_tokens.permute(0, 2, 3, 4, 1).reshape(1, new_d * new_h * new_w, embedding_size)

            # Concatenate extra tokens back
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

            print(f"Positional embeddings interpolated successfully. New shape: {new_pos_embed.shape}")


def interpolate_decoder_pos_embed(model, checkpoint_model):
    """Interpolate the decoder positional embeddings for MAE if the input size changes."""
    if 'decoder_pos_embed' in checkpoint_model:
        dec_pos_embed_checkpoint = checkpoint_model['decoder_pos_embed']
        embedding_size = dec_pos_embed_checkpoint.shape[-1]

        # The new model's decoder_pos_embed
        dec_pos_embed_model = model.decoder_pos_embed
        num_patches_dec = dec_pos_embed_model.shape[-2]  # new patch count + any extra tokens
        num_extra_tokens = dec_pos_embed_model.shape[-2] - model.patch_embed.num_patches

        # old patch count
        num_patches_old = dec_pos_embed_checkpoint.shape[1] - num_extra_tokens

        # For simplicity, assume cubic patches (if truly 3D).
        orig_size = int(round(num_patches_old ** (1 / 3)))
        new_size = int(round(model.patch_embed.num_patches ** (1 / 3)))

        if orig_size != new_size:
            print(f"Interpolating DECODER pos_embed from {orig_size}x{orig_size}x{orig_size} to {new_size}x{new_size}x{new_size}")

            # Extra tokens (CLS, etc.) in the decoder
            extra_tokens = dec_pos_embed_checkpoint[:, :num_extra_tokens]
            dec_pos_tokens = dec_pos_embed_checkpoint[:, num_extra_tokens:]

            # Reshape for 3D trilinear interpolation
            dec_pos_tokens = dec_pos_tokens.reshape(
                1, orig_size, orig_size, orig_size, embedding_size
            ).permute(0, 4, 1, 2, 3)

            dec_pos_tokens = F.interpolate(
                dec_pos_tokens,
                size=(new_size, new_size, new_size),
                mode='trilinear',
                align_corners=False
            )
            dec_pos_tokens = dec_pos_tokens.permute(0, 2, 3, 4, 1).reshape(
                1, new_size**3, embedding_size
            )

            # Concatenate the extra tokens back
            new_dec_pos_embed = torch.cat((extra_tokens, dec_pos_tokens), dim=1)
            checkpoint_model['decoder_pos_embed'] = new_dec_pos_embed
