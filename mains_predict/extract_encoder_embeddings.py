#!/usr/bin/env python3
"""Extract encoder embeddings from the 3D-MAE ViT model.

This is a prediction-style utility for exporting representations rather than
classification probabilities. 
"""

import argparse
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1] if THIS_FILE.parent.name == "mains_predict" else THIS_FILE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import models_vit
from util.pos_embed import interpolate_pos_embed


EMBEDDING_TYPES = ("class", "pooled", "concat", "pre_logits", "normalized_concat", "all")


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "Extract encoder embeddings from a 3D-MAE foundation/fine-tuned model"
    )

    parser.add_argument("--model", default="vit_large_patch16_yo", type=str)
    parser.add_argument("--input_size", default=256, type=int)
    parser.add_argument("--drop_path", default=0.0, type=float)
    parser.add_argument("--nb_classes", default=1, type=int)
    parser.add_argument("--finetune", required=True, type=str, help="Checkpoint path")
    parser.add_argument("--global_pool", action="store_true")
    parser.set_defaults(global_pool=True)
    parser.add_argument(
        "--cls_token",
        action="store_false",
        dest="global_pool",
        help="Build the model without global-pool head layers. Only use for matching old checkpoints.",
    )

    parser.add_argument("--input_csv", required=True, type=str)
    parser.add_argument("--output_csv", required=True, type=str)
    parser.add_argument(
        "--path_col",
        default="Path",
        type=str,
        help="CSV column containing image paths. Use 'Cases' for the old ILA NPZ CSVs.",
    )
    parser.add_argument(
        "--embedding_type",
        default="pre_logits",
        choices=EMBEDDING_TYPES,
        help=(
            "class=CLS token, pooled=mean patch token, concat=raw CLS+pooled, "
            "pre_logits/normalized_concat=fc_norm(raw CLS+pooled), all=write all four."
        ),
    )
    parser.add_argument("--embedding_col", default="Embedding", type=str)
    parser.add_argument(
        "--flatten_columns",
        action="store_true",
        help="Write one numeric CSV column per embedding dimension instead of a JSON list column.",
    )
    parser.add_argument(
        "--output_npy",
        default=None,
        type=str,
        help="Optional .npy/.npz sidecar containing the embedding matrix/matrices.",
    )

    parser.add_argument(
        "--volume_key",
        default="volume",
        type=str,
        help="Array key to read from .npz inputs.",
    )
    parser.add_argument(
        "--no_hu_normalize",
        action="store_true",
        help="For NIfTI inputs, skip clipping to [-1200, 800] and scaling to [0, 1].",
    )
    parser.add_argument(
        "--orientation_mod",
        action="store_true",
        help="After adding the channel dimension, permute volume as (C, W, H, D). Matches old ILA datasets.",
    )

    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=False)
    parser.add_argument("--amp", action="store_true", help="Use CUDA autocast during extraction")

    return parser


class VolumePathDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        path_col: str,
        volume_key: str = "volume",
        hu_normalize: bool = True,
        orientation_mod: bool = False,
    ) -> None:
        self.csv_path = csv_path
        self.path_col = path_col
        self.volume_key = volume_key
        self.hu_normalize = hu_normalize
        self.orientation_mod = orientation_mod
        self.samples = pd.read_csv(csv_path)

        if path_col not in self.samples.columns:
            raise ValueError(
                f"CSV path column '{path_col}' was not found. "
                f"Available columns: {self.samples.columns.tolist()}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, int]:
        file_path = str(self.samples.iloc[idx][self.path_col])
        volume = self._load_volume(file_path)

        volume_tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0)
        if self.orientation_mod:
            volume_tensor = volume_tensor.permute(0, 3, 2, 1)

        return volume_tensor, file_path, idx

    def _load_volume(self, file_path: str) -> np.ndarray:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")

        suffixes = "".join(Path(file_path).suffixes).lower()
        if suffixes.endswith(".npz"):
            with np.load(file_path) as data:
                if self.volume_key not in data:
                    raise KeyError(
                        f"'{self.volume_key}' not found in {file_path}. "
                        f"Available arrays: {list(data.keys())}"
                    )
                volume = data[self.volume_key].astype(np.float32)
        elif suffixes.endswith(".npy"):
            volume = np.load(file_path).astype(np.float32)
        else:
            try:
                import SimpleITK as sitk
            except ImportError as exc:
                raise ImportError("SimpleITK is required for NIfTI/DICOM-style inputs") from exc

            sitk_image = sitk.ReadImage(file_path)
            volume = sitk.GetArrayFromImage(sitk_image).astype(np.float32)
            if self.hu_normalize:
                volume = np.clip(volume, -1200, 800)
                volume = (volume + 1200) / 2000

        if volume.ndim != 3:
            raise ValueError(f"Expected a 3D volume at {file_path}, got shape {volume.shape}")
        return volume


def clean_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        cleaned[key] = value
    return cleaned


def get_checkpoint_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("model", "model_state", "state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return clean_state_dict_keys(checkpoint[key])
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint object: {type(checkpoint)}")
    return clean_state_dict_keys(checkpoint)


def load_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
        img_size=args.input_size,
    )

    checkpoint_model = get_checkpoint_state_dict(args.finetune)
    if "pos_embed" in checkpoint_model:
        interpolate_pos_embed(model, checkpoint_model)

    model_dict = model.state_dict()
    filtered_checkpoint = {
        key: value
        for key, value in checkpoint_model.items()
        if key in model_dict and model_dict[key].shape == value.shape
    }
    dropped = sorted(set(checkpoint_model) - set(filtered_checkpoint))

    msg = model.load_state_dict(filtered_checkpoint, strict=False)
    print(f"Loaded checkpoint: {args.finetune}")
    print(msg)
    if dropped:
        preview = ", ".join(dropped[:12])
        extra = "" if len(dropped) <= 12 else f", ... ({len(dropped)} total)"
        print(f"Dropped unmatched checkpoint keys: {preview}{extra}")

    model.to(device)
    model.eval()
    return model


def forward_encoder_tokens(model: torch.nn.Module, volume: torch.Tensor) -> torch.Tensor:
    """Return transformer tokens before final representation pooling.

    Shape is [B, 1 + num_patches, embed_dim], where token 0 is the CLS token.
    This mirrors models_vit.VisionTransformer.forward_features but keeps the
    patch sequence available so class/pooled/concat embeddings can be chosen.
    """

    batch_size = volume.shape[0]
    x = model.patch_embed(volume)
    cls_tokens = model.cls_token.expand(batch_size, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    x = x + model.pos_embed
    x = model.pos_drop(x)

    for block in model.blocks:
        x = block(x)

    return x


def embeddings_from_tokens(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    embedding_type: str,
) -> Dict[str, torch.Tensor]:
    cls_token = tokens[:, 0]
    pooled_patch_tokens = tokens[:, 1:, :].mean(dim=1)
    concat = torch.cat((cls_token, pooled_patch_tokens), dim=1)

    pre_logits = None
    if embedding_type in ("pre_logits", "normalized_concat", "all"):
        if not hasattr(model, "fc_norm"):
            raise AttributeError(
                "This model has no fc_norm layer. Use --global_pool, or choose "
                "--embedding_type class/pooled/concat."
            )
        pre_logits = model.fc_norm(concat)

    if embedding_type == "class":
        return {"class": cls_token}
    if embedding_type == "pooled":
        return {"pooled": pooled_patch_tokens}
    if embedding_type == "concat":
        return {"concat": concat}
    if embedding_type in ("pre_logits", "normalized_concat"):
        return {"pre_logits": pre_logits}

    return {
        "class": cls_token,
        "pooled": pooled_patch_tokens,
        "concat": concat,
        "pre_logits": pre_logits,
    }


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    embedding_type: str,
    use_amp: bool,
) -> Tuple[Dict[str, np.ndarray], List[str], List[int]]:
    outputs: Dict[str, List[np.ndarray]] = {}
    file_paths: List[str] = []
    row_indices: List[int] = []

    for volume, batch_paths, batch_indices in data_loader:
        volume = volume.to(device, non_blocking=True)
        autocast_context = (
            torch.cuda.amp.autocast(enabled=True)
            if device.type == "cuda" and use_amp
            else nullcontext()
        )
        with autocast_context:
            tokens = forward_encoder_tokens(model, volume)
            batch_embeddings = embeddings_from_tokens(model, tokens, embedding_type)

        for name, tensor in batch_embeddings.items():
            outputs.setdefault(name, []).append(tensor.detach().float().cpu().numpy())

        file_paths.extend(list(batch_paths))
        row_indices.extend([int(idx) for idx in batch_indices])

    stacked = {name: np.concatenate(chunks, axis=0) for name, chunks in outputs.items()}
    return stacked, file_paths, row_indices


def add_embeddings_to_dataframe(
    df: pd.DataFrame,
    embeddings: Dict[str, np.ndarray],
    row_indices: Iterable[int],
    embedding_col: str,
    flatten_columns: bool,
) -> pd.DataFrame:
    row_indices = list(row_indices)
    if len(embeddings) == 1:
        name, matrix = next(iter(embeddings.items()))
        col_base = embedding_col
        matrices = [(col_base, matrix)]
    else:
        matrices = [(f"{embedding_col}_{name}", matrix) for name, matrix in embeddings.items()]

    for col_base, matrix in matrices:
        if flatten_columns:
            for dim in range(matrix.shape[1]):
                df.loc[row_indices, f"{col_base}_{dim}"] = matrix[:, dim]
        else:
            df.loc[row_indices, col_base] = [json.dumps(row.tolist()) for row in matrix]

    return df


def save_sidecar(output_npy: str, embeddings: Dict[str, np.ndarray]) -> None:
    output_path = Path(output_npy)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(embeddings) == 1 and output_path.suffix.lower() == ".npy":
        np.save(output_path, next(iter(embeddings.values())))
    else:
        np.savez_compressed(output_path, **embeddings)
    print(f"Embedding sidecar saved to {output_path}")


def main(args: argparse.Namespace) -> None:
    requested_device = torch.device(args.device)
    device = requested_device if requested_device.type != "cuda" or torch.cuda.is_available() else torch.device("cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    model = load_model(args, device)

    dataset = VolumePathDataset(
        csv_path=args.input_csv,
        path_col=args.path_col,
        volume_key=args.volume_key,
        hu_normalize=not args.no_hu_normalize,
        orientation_mod=args.orientation_mod,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )

    embedding_type = "pre_logits" if args.embedding_type == "normalized_concat" else args.embedding_type
    embeddings, file_paths, row_indices = extract_embeddings(
        model=model,
        data_loader=data_loader,
        device=device,
        embedding_type=embedding_type,
        use_amp=args.amp,
    )

    for name, matrix in embeddings.items():
        print(f"{name}: {matrix.shape}")

    df = pd.read_csv(args.input_csv)
    df = add_embeddings_to_dataframe(
        df=df,
        embeddings=embeddings,
        row_indices=row_indices,
        embedding_col=args.embedding_col,
        flatten_columns=args.flatten_columns,
    )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Embeddings saved to {output_csv}")

    if args.output_npy:
        save_sidecar(args.output_npy, embeddings)


if __name__ == "__main__":
    main(get_args_parser().parse_args())
