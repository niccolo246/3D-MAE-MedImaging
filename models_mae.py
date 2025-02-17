# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from patch_three_d import PatchEmbedThreeD

from util.pos_embed import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """
    Masked Autoencoder with Vision Transformer backbone for 3D data.

    This class implements a 3D extension of the MAE (Masked Autoencoder) framework,
    originally designed for 2D images. It divides a 3D volume into patches, encodes
    them using a Vision Transformer encoder, and then reconstructs masked patches via
    a Transformer decoder.

    Attributes:
        three_d (bool): Flag indicating whether to use 3D patch embedding.
        patch_embed (nn.Module): Patch embedding module (3D or 2D variant).
        cls_token (nn.Parameter): Learnable class token.
        pos_embed (nn.Parameter): Fixed positional embedding (initialized using sin-cos embeddings).
        blocks (nn.ModuleList): Transformer encoder blocks.
        norm (nn.Module): Layer normalization applied after the encoder.
        decoder_embed (nn.Linear): Linear layer projecting encoder output to decoder embedding space.
        mask_token (nn.Parameter): Learnable token representing masked patches.
        decoder_pos_embed (nn.Parameter): Fixed positional embeddings for the decoder.
        decoder_blocks (nn.ModuleList): Transformer decoder blocks.
        decoder_norm (nn.Module): Normalization layer applied after the decoder.
        decoder_pred (nn.Sequential): Prediction head that maps decoder output to reconstructed patch pixels.
        norm_pix_loss (bool): Flag to indicate if the reconstruction loss should be computed on normalized pixels.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, three_d=True):
        """
        Initializes the MaskedAutoencoderViT model.

        Args:
            img_size (int, optional): Input image size (assumed cubic). Defaults to 224.
            patch_size (int, optional): Size of each patch. Defaults to 16.
            in_chans (int, optional): Number of input channels. Defaults to 3.
            embed_dim (int, optional): Embedding dimension for the encoder. Defaults to 1024.
            depth (int, optional): Number of Transformer encoder blocks. Defaults to 24.
            num_heads (int, optional): Number of attention heads in the encoder. Defaults to 16.
            decoder_embed_dim (int, optional): Embedding dimension for the decoder. Defaults to 512.
            decoder_depth (int, optional): Number of Transformer decoder blocks. Defaults to 8.
            decoder_num_heads (int, optional): Number of attention heads in the decoder. Defaults to 16.
            mlp_ratio (float, optional): Ratio for the MLP expansion in Transformer blocks. Defaults to 4.
            norm_layer (Callable, optional): Normalization layer constructor (e.g., nn.LayerNorm). Defaults to nn.LayerNorm.
            norm_pix_loss (bool, optional): If True, uses normalized pixels for the reconstruction loss. Defaults to False.
            three_d (bool, optional): If True, uses a 3D patch embedding module; otherwise, uses the 2D version.
        """
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.three_d = three_d
        self.patch_embed = PatchEmbedThreeD(img_size, patch_size, in_chans, embed_dim) if self.three_d \
            else PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)  # qk_scale=None
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) # qk_scale=None
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.decoder_pred = nn.Sequential(
            nn.Linear(decoder_embed_dim, patch_size**3 * in_chans, bias=True),
            nn.Sigmoid()
        )
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize and freeze the positional embeddings using sin-cos functions.
        Also initializes the projection weights for the patch embedding and the tokens.
        """
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True) if self.three_d \
            else get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True)

        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True) if self.three_d \
            else get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv3d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Custom weight initialization for Linear and LayerNorm layers.
        """
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, volumes):
        """
        Converts a batch of 3D volumes into a sequence of flattened patches.
        Args:
            volumes (torch.Tensor): Input tensor of shape [N, C, D, H, W].
        Returns:
            torch.Tensor: Output tensor of shape [N, L, patch_size**3 * C], where L is the number of patches.
        """

        p = self.patch_embed.patch_size[0]
        assert volumes.shape[2] % p == 0 and volumes.shape[3] % p == 0 and volumes.shape[4] % p == 0

        d = volumes.shape[2] // p
        h = volumes.shape[3] // p
        w = volumes.shape[4] // p
        x = volumes.reshape(shape=(volumes.shape[0], volumes.shape[1], d, p, h, p, w, p))
        x = torch.einsum('ncdphqwr->ndhwpqrc', x)
        x = x.reshape(shape=(volumes.shape[0], d * h * w, p**3 * volumes.shape[1]))
        return x

    def unpatchify(self, x):
        """
        Reconstructs a batch of 3D volumes from a sequence of flattened patches.
        Args:
            x (torch.Tensor): Input tensor of shape [N, L, patch_size**3 * C].
        Returns:
            torch.Tensor: Reconstructed tensor of shape [N, C, D, H, W].
        """
        p = self.patch_embed.patch_size[0]
        N, L, C_p3 = x.shape
        C = C_p3 // (p**3)
        dhw = round((L) ** (1/3))  # Assuming the volume is cubic
        assert dhw * dhw * dhw == L

        x = x.reshape(shape=(N, dhw, dhw, dhw, p, p, p, C))
        x = torch.einsum('ndhwpqrc->ncdphqwr', x)
        volumes = x.reshape(shape=(N, C, dhw * p, dhw * p, dhw * p))
        return volumes

    def random_masking(self, x, mask_ratio):
        """
        Performs per-sample random masking of patches.
        The function shuffles patches using random noise, then keeps a proportion (1 - mask_ratio)
        of the patches while generating a binary mask indicating which patches were removed.
        Args:
            x (torch.Tensor): Input tensor of shape [N, L, D] (sequence of patch embeddings).
            mask_ratio (float): Ratio of patches to mask (i.e., remove).
        Returns:
            tuple: (x_masked, mask, ids_restore)
                - x_masked (torch.Tensor): Masked sequence of patches.
                - mask (torch.Tensor): Binary mask of shape [N, L] (0 for kept, 1 for removed patches).
                - ids_restore (torch.Tensor): Indices to restore the original order.
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        """
        Runs the encoder of the MAE.
        The function embeds patches, adds positional embeddings (excluding the class token),
        applies random masking, concatenates the class token, and passes the result through
        a series of Transformer encoder blocks followed by normalization.
        Args:
            x (torch.Tensor): Input tensor of shape [N, C, D, H, W].
            mask_ratio (float): The ratio of patches to mask.
        Returns:
            tuple: (x, mask, ids_restore)
                - x (torch.Tensor): The encoded representations including the class token.
                - mask (torch.Tensor): The binary mask used for random masking.
                - ids_restore (torch.Tensor): Indices to restore the original patch order.
        """
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        """
        Runs the decoder of the MAE.
        The function projects the encoder output into the decoder embedding space,
        appends mask tokens to the sequence, restores the original patch order, adds
        decoder positional embeddings, and passes the sequence through the decoder blocks.
        Finally, it produces the predicted pixel values for each patch and removes the class token.
        Args:
            x (torch.Tensor): Encoder output (excluding mask tokens) of shape [N, L', embed_dim].
            ids_restore (torch.Tensor): Indices to restore the original patch order.
        Returns:
            torch.Tensor: The predicted patch pixels of shape [N, L, patch_size**3 * C].
        """
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        Computes the reconstruction loss for the MAE.
        The function first converts the input 3D volumes into patches using `patchify`,
        optionally normalizes the patches, computes the mean squared error (MSE) loss
        between the predictions and the target patches, and finally computes the mean
        loss only over the masked patches.
        Args:
            imgs (torch.Tensor): Input volumes of shape [N, C, D, H, W].
            pred (torch.Tensor): Predicted patch values of shape [N, L, patch_size**3 * C].
            mask (torch.Tensor): Binary mask of shape [N, L] (0 for kept patches, 1 for masked patches).
        Returns:
            torch.Tensor: The computed reconstruction loss.
        """
        torch.set_printoptions(precision=6)
        target = self.patchify(imgs)  # Adjust patchify to handle 3D volumes

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        """
        Forward pass for the MAE.
        This function runs the encoder and decoder and computes the reconstruction loss.
        Args:
            imgs (torch.Tensor): Input volumes of shape [N, C, D, H, W].
            mask_ratio (float, optional): The ratio of patches to mask. Defaults to 0.75.
        Returns:
            tuple: (loss, pred, mask)
                - loss (torch.Tensor): The reconstruction loss.
                - pred (torch.Tensor): The predicted patch values.
                - mask (torch.Tensor): The binary mask indicating which patches were masked.
        """
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(img_size=256, in_chans=1,
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(img_size=256, in_chans=1,
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(img_size=256, in_chans=1,
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks




###########################################
#
#
# Reconstruction:
#
#

def mae_vit_large_patch16_reconstruct(**kwargs):
    model = MaskedAutoencoderViT(img_size=96, in_chans=1,
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

