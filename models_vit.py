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

import timm.models.vision_transformer

from patch_three_d import PatchEmbedThreeD
from util.pos_embed import get_3d_sincos_pos_embed


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """
    Vision Transformer with support for global pooling, adapted for 3D data.

    This implementation extends the standard Vision Transformer (ViT) by enabling a global
    pooling mode that leverages both the class token and the average-pooled patch tokens
    for the final prediction. When global_pool is enabled, the model concatenates the class
    token and the mean of the patch tokens, passes them through an additional normalization
    layer, and then uses a modified head to produce the final predictions.

    Attributes:
        global_pool (bool): If True, global pooling is applied using both the class token and
            the average of patch tokens.
        fc_norm (nn.Module): A normalization layer applied to the concatenated features when
            global pooling is enabled.
        head (nn.Module): The classification head that accepts the concatenated features.
        patch_embed (nn.Module): The 3D patch embedding module (PatchEmbedThreeD) used to divide
            the input volume into patches.
        pos_embed (nn.Parameter): Fixed (sin-cos) positional embeddings for the patches, including
            the class token.
    """
    def __init__(self, global_pool=False, patch_size=16, embed_dim=1024, img_size=256, **kwargs):
        """
        Initializes the VisionTransformer model for 3D data.
        Args:
            global_pool (bool): If True, enables global pooling that concatenates the class token
                with the average-pooled patch tokens for prediction.
            patch_size (int): The size of each patch.
            embed_dim (int): The embedding dimension.
            **kwargs: Additional keyword arguments passed to the parent ViT class, which should
                include at least 'norm_layer' and 'num_classes'.
        """
        kwargs['embed_dim'] = embed_dim
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim*2)  # modification ¢¢¢¢¢¢¢

            num_classes = kwargs['num_classes']
            self.head = nn.Linear(self.embed_dim * 2, num_classes) if num_classes > 0 else nn.Identity()  # modification ¢¢¢¢¢¢¢

            del self.norm  # remove the original norm

        self.patch_embed = PatchEmbedThreeD(img_size=img_size, patch_size=patch_size, in_chans=1, embed_dim=embed_dim)  # hard coded for vit large
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward_features(self, x):
        """
        Computes the feature representations from the input volume.

        This method embeds the input volume into patches, adds positional embeddings (including
        the class token), and passes the resulting sequence through the Transformer encoder blocks.
        In global pooling mode, it computes the mean of the patch tokens (excluding the class token)
        and concatenates this with the class token; otherwise, it uses the class token directly.
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, D, H, W].
        Returns:
            torch.Tensor: Feature representation used for classification.
                - If global_pool is True, the output shape is [B, 2 * embed_dim].
                - Otherwise, the output is the class token of shape [B, embed_dim].
        """
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            cls_token = x[:, 0]  # Select the class token
            patch_tokens = x[:, 1:, :].mean(dim=1)  # Apply global pooling (excluding class token)
            combined = torch.cat((cls_token, patch_tokens), dim=1)  # Combine class token and pooled patches
            outcome = combined
        else:
            x = self.norm(x)

            outcome = x[:, 0]

        return outcome

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        """
        Processes the features to produce final predictions.
        This method applies an attention pooling if available or, if global_pool equals 'avg', it
        averages over the tokens (excluding any prefix tokens). Then it normalizes the resulting features
        with fc_norm and passes them through the classification head.
        Args:
            x (torch.Tensor): Input features from the encoder.
            pre_logits (bool): If True, returns the normalized features before the head layer.
        Returns:
            torch.Tensor: The final logits or pre-logits based on the pre_logits flag.
        """
        if self.attn_pool is not None:
            x = self.attn_pool(x)
        elif self.global_pool == 'avg':
            x = x[:, self.num_prefix_tokens:].mean(dim=1)

        x = self.fc_norm(x)

        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a full forward pass of the MAE model.
        This method combines the encoder and the head to compute the final prediction.
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, D, H, W].
        Returns:
            torch.Tensor: The final output logits.
        """
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class VisionTransformerMod(timm.models.vision_transformer.VisionTransformer):
    """
    Vision Transformer with Global Pooling and a Custom Multi-Layer Perceptron (MLP) Head.

    This variant of the Vision Transformer extends the standard ViT by:
      - Utilizing 3D patch embedding (via PatchEmbedThreeD) for volumetric data.
      - Enabling global pooling that combines the class token with the average-pooled patch tokens.
      - Replacing the original classification head with a custom MLP head composed of multiple hidden layers.

    In global pooling mode, the final feature representation is obtained by concatenating the class token
    with the mean-pooled patch tokens, doubling the feature dimension before being passed to the MLP head.
    """
    def __init__(self, global_pool=False, patch_size=16, embed_dim=1024, hidden_layers=[64, 32], **kwargs):
        """
        Initializes the VisionTransformerMod.
        Args:
            global_pool (bool): If True, enables global pooling that concatenates the class token
                                with the mean-pooled patch tokens for the final prediction.
            patch_size (int): The size of each patch (used in the 3D patch embedding).
            embed_dim (int): The embedding dimension for the encoder.
            hidden_layers (list[int]): A list of integers specifying the sizes of hidden layers in the custom MLP head.
            **kwargs: Additional keyword arguments for the parent VisionTransformer. Must include at least:
                      - norm_layer: A callable that returns a normalization layer (e.g., nn.LayerNorm).
                      - num_classes: The number of output classes.
        """
        kwargs['embed_dim'] = embed_dim
        super(VisionTransformerMod, self).__init__(**kwargs)

        self.global_pool = global_pool
        norm_layer = kwargs['norm_layer']
        embed_dim = kwargs['embed_dim']
        num_classes = kwargs['num_classes']

        # fc_norm normalization layer
        self.fc_norm = norm_layer(embed_dim * 2)  # Combining cls token and pooled patch tokens

        # Build the MLP head with hidden layers [256, 128, 64]
        # Input dimension = embed_dim * 2, Output dimension = num_classes
        mlp_layers = []
        input_dim = embed_dim * 2
        for h in hidden_layers:
            mlp_layers.append(nn.Linear(input_dim, h))
            mlp_layers.append(nn.LeakyReLU())
            # Optionally add BatchNorm/Dropout if desired:
            mlp_layers.append(nn.BatchNorm1d(h))
            # mlp_layers.append(nn.Dropout(0.5))
            input_dim = h

        mlp_layers.append(nn.Linear(input_dim, num_classes))
        self.head = nn.Sequential(*mlp_layers)

        # Remove the original norm if we have global_pool
        if self.global_pool:
            del self.norm

        # Use the provided 3D patch embedding
        self.patch_embed = PatchEmbedThreeD(img_size=img_size, patch_size=patch_size, in_chans=1, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward_features(self, x):
        """
        Extracts feature representations from the input volume.
        The input is passed through the 3D patch embedding, followed by the addition of
        fixed positional embeddings. The resulting sequence is processed by Transformer blocks.
        When global pooling is enabled, the class token is concatenated with the average-pooled
        patch tokens.
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, D, H, W].
        Returns:
            torch.Tensor: The resulting feature representation.
                - If global_pool is True: [B, embed_dim * 2]
                - Otherwise: [B, embed_dim] (using only the class token)
        """
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            # Separate the class token and apply global pooling to patch tokens
            cls_token = x[:, 0]  # class token
            patch_tokens = x[:, 1:, :].mean(dim=1)  # global avg pool on patch tokens
            combined = torch.cat((cls_token, patch_tokens), dim=1)  # (B, embed_dim*2)
            outcome = combined
        else:
            x = self.norm(x)
            outcome = x[:, 0]  # CLS token if not global pooling

        return outcome

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        """
        Processes the feature representation through the custom MLP head to produce predictions.
        The feature vector is first normalized by fc_norm. If pre_logits is True, the normalized
        feature is returned without further processing.
        Args:
            x (torch.Tensor): Feature representation from the encoder.
            pre_logits (bool): If True, return the normalized features before applying the head.
        Returns:
            torch.Tensor: Final logits if pre_logits is False; otherwise, the normalized features.
        """
        x = self.fc_norm(x)
        if pre_logits:
            return x
        return self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"
        Performs a full forward pass through the network.
        Combines the extraction of features and the custom head to produce the final prediction logits.
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, D, H, W].
        Returns:
            torch.Tensor: The output logits.
        """
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16_yo(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_large_patch16_power_2_yo(**kwargs):
    model = VisionTransformerMod(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), hidden_layers=[64], **kwargs)
    return model

def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16_reg(**kwargs):
    model = VisionTransformer(regression=True,
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model





#################################
#
# Prediction example:
#
#
#################################


# Create the model instance
#model = vit_large_patch16_yo(img_size=96, num_classes=1, global_pool=True)

# Define an input tensor with shape (1,1,96,96,96)
#input_tensor = torch.randn(1, 1, 96, 96, 96)

# Forward pass
#output = model(input_tensor)

# Print output shape
#print(f"Input size: {input_tensor.shape}")
#print(f"Output size: {output.shape}")





