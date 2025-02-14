# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import numpy as np
import os
from pathlib import Path
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import util.misc as misc
import models_vit
from datasets_three_d_fine import Custom3DDataset


def get_args_parser():
    parser = argparse.ArgumentParser(
        'MAE fine-tuning prediction for image classification', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16_power_3_yo', type=str,
                        metavar='MODEL', help='Name of model to load')
    parser.add_argument('--input_size', default=256, type=int,
                        help='Image input size')
    parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                        help='Drop path rate')

    # Finetuning parameter
    parser.add_argument('--finetune', default='/path/to/best_model_loss.pth',
                        help='Path to pre-trained checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool')

    # Dataset / CSV parameters
    parser.add_argument('--input_csv', default='/path/to/input.csv', type=str,
                        help='Path to input CSV with dataset information')
    parser.add_argument('--output_csv', default='/path/to/output_predictions.csv', type=str,
                        help='Path to output CSV for predictions')
    parser.add_argument('--nb_classes', default=1, type=int,
                        help='Number of classes')
    parser.add_argument('--regression', default=False, type=int,
                        help='Flag for regression task')
    parser.add_argument('--binary_class', default=True, type=int,
                        help='Flag for binary classification')
    parser.add_argument('--binary_class_weights', default=None, type=float, nargs='+',
                        help='Weights for binary classification (if applicable)')

    # General parameters
    parser.add_argument('--device', default='cuda', help='Device to use for prediction')
    parser.add_argument('--seed', default=0, type=int)

    # DataLoader parameters for prediction
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size for prediction')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Number of DataLoader workers')
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=False)

    # Distributed training parameters (if needed)
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='URL used to set up distributed training')

    return parser


def main(args):
    # Initialize distributed mode (if applicable)
    misc.init_distributed_mode(args)
    print('Job dir:', os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n", str(args).replace(', ', ',\n'))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Set seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Setup criterion (used only to decide activation in prediction)
    if args.binary_class:
        if args.binary_class_weights:
            weights = torch.tensor(args.binary_class_weights)
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
        else:
            criterion = torch.nn.BCEWithLogitsLoss()
    elif args.regression:
        criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # Load the model
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    # Load pre-trained checkpoint
    checkpoint_model = torch.load(args.finetune, map_location='cpu')
    print("Loading checkpoint from:", args.finetune)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model:", model_without_ddp)
    print('Number of params (M): %.2f' % (n_parameters / 1.e6))

    # Wrap model with DistributedDataParallel if running distributed prediction
    if args.world_size > 1:
        args.gpu = args.local_rank  # alias local_rank to gpu for compatibility
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    @torch.no_grad()
    def predict(model, data_loader, device):
        model.eval()
        predictions = []
        file_paths = []
        for volume, file_path in data_loader:
            volume = volume.to(device, non_blocking=True)
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    output = model(volume)
            else:
                output = model(volume)
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                pred = torch.softmax(output, dim=1)
            elif isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                pred = torch.sigmoid(output)
            else:
                raise ValueError("Unsupported criterion type")
            predictions.extend(pred.cpu().numpy().tolist())
            file_paths.extend(file_path)
        return predictions, file_paths

    # Load the dataset for prediction
    dataset = Custom3DDataset(args.input_csv)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_mem)

    # Perform predictions
    predictions, file_paths = predict(model, data_loader, device)

    # Load the original CSV and add predictions
    df = pd.read_csv(args.input_csv)
    df['Predictions'] = pd.Series(predictions)
    df.to_csv(args.output_csv, index=False)
    print("Predictions saved to", args.output_csv)


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)






