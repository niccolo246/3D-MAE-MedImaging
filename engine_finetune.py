# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup

import util.misc as misc
import util.lr_sched as lr_sched

import numpy as np

from sklearn.metrics import f1_score, roc_auc_score
import os


def train_one_epoch_orig(
        model: torch.nn.Module, criterion: torch.nn.Module,
        data_loader: Iterable, optimizer: torch.optim.Optimizer,
        device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
        mixup_fn: Optional[Mixup] = None, log_writer=None, args=None
        ):
    """
    Train the model for one epoch with gradient accumulation, learning rate scheduling,
    and optional mixup augmentation, using mixed precision (if enabled).

    This function performs a single epoch of training. It iterates over the given data_loader,
    updates the learning rate per iteration, and applies gradient accumulation (controlled by args.accum_iter).
    Mixed precision training is enabled via torch.cuda.amp.autocast(), and if a mixup function is provided,
    it is applied to the input samples and targets. Training metrics (loss and learning rate) are logged
    using a MetricLogger and optionally recorded via a TensorBoard log writer.

    Parameters:
        model (torch.nn.Module): The model to train.
        criterion (torch.nn.Module): The loss function.
        data_loader (Iterable): An iterable DataLoader that yields (samples, targets).
        optimizer (torch.optim.Optimizer): The optimizer for updating model weights.
        device (torch.device): The device on which to perform training.
        epoch (int): The current epoch number (used for logging and scheduling).
        loss_scaler: A function or object that scales the loss for mixed precision training.
        max_norm (float, optional): Maximum gradient norm for clipping. Defaults to 0 (no clipping).
        mixup_fn (Optional[Mixup], optional): A mixup function to apply on the samples and targets. Defaults to None.
        log_writer (optional): A TensorBoard SummaryWriter to log training metrics.
        args: Additional arguments (must contain at least 'accum_iter' for gradient accumulation,
              and may include other training settings).

    Returns:
        dict: A dictionary mapping metric names to their global average values computed over the epoch.

    Raises:
        SystemExit: If the computed loss is not finite, the training is terminated.
    """

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    total_loss = 0.0
    total_samples = 0

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            optimizer.zero_grad()
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)

            if outputs.shape != targets.shape:
                outputs = outputs.squeeze(-1)  # Ensure shape consistency

            loss = criterion(outputs, targets)

        loss_value = loss.item()
        batch_size = samples.size(0)
        total_loss += loss_value * batch_size
        total_samples += batch_size

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # Note: No scaling here
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)  # Update with the true loss value
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ #We use epoch_1000x as the x-axis in tensorboard.
            'This calibrates different curves when batch size changes.'
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # Calculate the average loss over the epoch
    if total_samples > 0:  # Ensure no division by zero
        avg_loss = total_loss / total_samples
        metric_logger.meters['loss'].update(avg_loss)

    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_iter(
        model: torch.nn.Module, criterion: torch.nn.Module,
        data_loader: Iterable, optimizer: torch.optim.Optimizer,
        device: torch.device, epoch: int, loss_scaler,
        max_norm: float = 0, mixup_fn: Optional[Mixup] = None,
        log_writer=None, args=None, iteration_counter=0,
        validate_every=20, data_loader_val=None, best_metrics=None, output_dir=None
        ):
    """
    Trains the model for one epoch using an iterative loop that supports periodic validation.

    This function performs training on the provided data_loader with gradient accumulation
    and mixed-precision training (using torch.cuda.amp.autocast()). It adjusts the learning
    rate on a per-iteration basis and supports optional mixup augmentation. In addition to
    training, the function performs validation every `validate_every` iterations using the
    provided validation data_loader. If the validation metrics improve (e.g., lower loss or
    higher AUC), the model is saved to the specified output directory. The function returns
    aggregated training metrics, the best validation metrics observed, and the most recent
    test statistics.

    Args:
        model (torch.nn.Module): The model to be trained.
        criterion (torch.nn.Module): The loss function.
        data_loader (Iterable): The training data loader that yields (samples, targets).
        optimizer (torch.optim.Optimizer): The optimizer used for updating model weights.
        device (torch.device): The device (e.g., 'cuda' or 'cpu') on which to run training.
        epoch (int): The current epoch number (used for scheduling and logging).
        loss_scaler: A loss scaling utility for mixed precision training.
        max_norm (float, optional): Maximum gradient norm for clipping. Defaults to 0 (no clipping).
        mixup_fn (Optional[Mixup], optional): A function for applying mixup augmentation. Defaults to None.
        log_writer: An optional TensorBoard log writer for logging metrics.
        args: Additional arguments, which must include at least 'accum_iter' (gradient accumulation iterations)
              and may include other training parameters.
        iteration_counter (int, optional): The starting iteration count (useful if resuming training). Defaults to 0.
        validate_every (int, optional): Frequency (in iterations) at which to perform validation. Defaults to 20.
        data_loader_val (Optional[Iterable], optional): Validation data loader. If None, validation is skipped.
        best_metrics (dict, optional): A dictionary holding the best validation metrics observed so far.
                                       Expected keys include "best_val_loss" and "best_auc". Defaults to None.
        output_dir (str, optional): Directory where model checkpoints will be saved when validation metrics improve.

    Returns:
        tuple: A tuple containing:
            - metric_logger (misc.MetricLogger): Aggregated training metrics over the epoch.
            - best_metrics (dict): Updated dictionary with the best validation metrics.
            - last_test_stats (dict): The validation statistics from the most recent validation step.

    Raises:
        SystemExit: If the computed loss is not finite.
    """

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 20

    # Initialize a variable to store the latest test_stats
    last_test_stats = {}

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # Adjust learning rate and zero gradients
        if (iteration_counter + data_iter_step) % args.accum_iter == 0:
            optimizer.zero_grad()
            current_epoch = epoch + data_iter_step / len(data_loader)
            lr_sched.adjust_learning_rate(optimizer, current_epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            if outputs.shape != targets.shape:
                outputs = outputs.squeeze(-1)
            loss = criterion(outputs, targets)

        # Update the loss scaler and perform optimizer step
        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=False)

        # **Update the metric logger**
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # Perform validation every `validate_every` iterations
        current_step = iteration_counter + data_iter_step
        if (current_step % validate_every == 0) and (current_step > 0) and (data_loader_val is not None):
            model.eval()
            with torch.no_grad():
                test_stats = evaluate(data_loader_val, model, device, criterion)

            val_loss = test_stats['loss']
            val_auc = test_stats.get('auc', 0.0)
            val_accuracy = test_stats.get('accuracy', 0.0)
            print(f"Iteration {current_step}: "
                  f"Validation loss: {val_loss:.4f}, AUC: {val_auc:.4f}, Accuracy: {val_accuracy:.4f}")

            # Update best metrics and save model
            if val_loss < best_metrics["best_val_loss"]:
                best_metrics["best_val_loss"] = val_loss
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model_loss.pth'))
                print(f"New best model saved with validation loss: {val_loss:.4f}")

            if val_auc > best_metrics["best_auc"]:
                best_metrics["best_auc"] = val_auc
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model_auc.pth'))
                print(f"New best model saved with validation AUC: {val_auc:.4f}")

            model.train(True)

            # **Store the latest test_stats**
            last_test_stats = test_stats  # Keep the latest test_stats for logging

    return metric_logger, best_metrics, last_test_stats


@torch.no_grad()
def evaluate(data_loader, model, device, criterion):
    """
    Evaluate the model on the given data loader and compute performance metrics.

    This function performs evaluation in no-gradient mode, optionally using mixed precision.
    It computes the average loss over the dataset and, for classification tasks, calculates
    accuracy, F1 score, and AUC. For regression tasks, it computes the mean squared error (MSE).

    Args:
        data_loader (Iterable): DataLoader that yields batches of (images, target).
        model (torch.nn.Module): The model to evaluate.
        device (torch.device): The device on which evaluation is performed.
        criterion (torch.nn.Module): The loss function. Expected to be one of:
            - torch.nn.CrossEntropyLoss
            - torch.nn.BCEWithLogitsLoss
            - torch.nn.MSELoss

    Returns:
        dict: A dictionary containing evaluation metrics. For classification, keys include
              'loss', 'accuracy', 'f1', and 'auc'. For regression, key 'mse' is provided along with 'loss'.
    """

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    model.eval()  # Set model to evaluation mode

    total_loss = 0.0
    total_samples = 0
    total_mse = 0.0  # Only used for MSE loss

    all_preds = []
    all_targets = []
    all_probs = []  # To store probabilities for AUC

    is_classification = isinstance(criterion, (torch.nn.CrossEntropyLoss, torch.nn.BCEWithLogitsLoss))

    for data_iter_step, (images, target) in enumerate(data_loader):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if isinstance(criterion, torch.nn.MSELoss):
            target = target.float()

        with torch.cuda.amp.autocast():
            output = model(images)

            if output.shape != target.shape:
                output = output.squeeze(-1)  # Ensure shape consistency

            loss = criterion(output, target)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if is_classification:
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                pred = torch.argmax(output, dim=1)
                probs = torch.softmax(output, dim=1)[:, 1]  # For AUC, we need the probability of class 1
            elif isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                probs = torch.sigmoid(output)  # Get probabilities for binary classification
                pred = torch.round(probs)  # Apply threshold at 0.5 for predictions

            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())
            all_probs.append(probs.cpu())  # Store probabilities for AUC calculation
        else:
            mse = torch.mean((output - target) ** 2)
            total_mse += mse.item() * batch_size

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

    if is_classification:
        all_preds = torch.cat(all_preds).numpy().astype(np.float32)
        all_targets = torch.cat(all_targets).numpy().astype(np.float32)
        all_probs = torch.cat(all_probs).numpy().astype(np.float32)

        print(all_preds)
        print(all_targets)

        if isinstance(criterion, torch.nn.CrossEntropyLoss):
            accuracy = (all_preds == all_targets).mean()
            f1 = f1_score(all_targets, all_preds, average='weighted')  # Use weighted for multi-class
            auc = roc_auc_score(all_targets, all_probs, multi_class='ovr')  # AUC for multi-class
        elif isinstance(criterion, torch.nn.BCEWithLogitsLoss):
            if all_targets.ndim > 1 and all_targets.shape[1] > 1:  # Multi-label
                accuracy = (all_preds == all_targets).mean()
                f1 = f1_score(all_targets, all_preds, average='samples')
                auc = roc_auc_score(all_targets, all_probs, average='macro')  # Multi-label AUC
            else:  # Binary
                accuracy = (all_preds == all_targets).mean()
                f1 = f1_score(all_targets, all_preds, average='binary')
                auc = roc_auc_score(all_targets, all_probs)  # AUC for binary

    metrics = {'loss': avg_loss}
    if is_classification:
        metrics['accuracy'] = accuracy
        metrics['f1'] = f1
        metrics['auc'] = auc
        print('* Accuracy {accuracy:.3f} F1 {f1:.3f} AUC {auc:.3f} loss {avg_loss:.3f}'.format(accuracy=accuracy, f1=f1, auc=auc, avg_loss=avg_loss))
    else:
        avg_mse = total_mse / total_samples if total_samples > 0 else 0.0
        metrics['mse'] = avg_mse
        print('* MSE {mse:.3f} loss {avg_loss:.3f}'.format(mse=avg_mse, avg_loss=avg_loss))

    return metrics
