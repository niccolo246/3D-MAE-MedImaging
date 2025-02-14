# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import os
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk


class Custom3DDataset(Dataset):
    """
    A custom dataset for loading 3D medical images (e.g., CT scans) stored in NIfTI format using SimpleITK,
    with associated labels provided in a CSV file.

    The CSV file should contain at least two columns:
        - 'Path': the file path to the NIfTI image.
        - 'Label': the corresponding label for the image.

    Each image is read with SimpleITK, converted to a numpy array, clamped to a specified range of Hounsfield Units,
    normalized to [0, 1] using fixed min and max values, and then converted to a torch tensor with an added channel dimension.
    Note that SimpleITK returns the image array in [Depth, Height, Width] order.
    """
    def __init__(self, csv_path: str, transform: Optional[callable] = None) -> None:
        """
        Args:
            csv_path (str): Path to the CSV file containing image paths and labels.
            transform (callable, optional): A function/transform to apply to the volume.
        """
        self.transform = transform
        self.samples = self._load_samples(csv_path)

    def _load_samples(self, csv_path: str) -> pd.DataFrame:
        """
        Loads sample data from a CSV file.

        Args:
            csv_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: DataFrame containing the sample data.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        data = pd.read_csv(csv_path)
        required_columns = {'Path', 'Label'}
        if not required_columns.issubset(data.columns):
            raise ValueError(f"CSV file must contain columns {required_columns}. Found columns: {data.columns.tolist()}")
        return data

    def __len__(self) -> int:
        """
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the volume and label corresponding to the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - volume_tensor: The 3D volume as a torch tensor with shape [1, D, H, W].
                - label: The corresponding label as a torch tensor.
        """
        # Get file path and label from the CSV DataFrame.
        file_path = self.samples.iloc[idx]['Path']
        label = self.samples.iloc[idx]['Label']

        # Check if the image file exists.
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")

        # Load the image using SimpleITK
        try:
            sitk_image = sitk.ReadImage(file_path)
        except Exception as e:
            print(f"Error loading image at {file_path}: {e}")
            raise

        # Get the numpy array from the image.
        # SimpleITK returns an array with shape (Depth, Height, Width).
        volume = sitk.GetArrayFromImage(sitk_image).astype(np.float32)

        # Clamp the values between -1200 and 800 Hounsfield Units.
        volume = np.clip(volume, -1200, 800)

        # Min-max normalization to [0, 1]
        volume = (volume + 1200) / (800 + 1200)

        # Convert to torch tensor and add channel dimension.
        volume_tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0)  # Shape: [1, D, H, W]

        if self.transform:
            volume_tensor = self.transform(volume_tensor)

        return volume_tensor, torch.tensor(label, dtype=torch.float32)


# Optional utility: Use resample_ct_volume() to resample CT volumes to a consistent size.
def resample_ct_volume(input_image_path: str, output_image_path: str, new_size: tuple = (256, 256, 256)) -> None:
    """
    Resample a CT volume (in .nii.gz format) to a specified size.

    This function reads a CT volume from the input path, computes the new voxel spacing
    based on the desired output size (to preserve the physical dimensions), and resamples
    the image using BSpline interpolation. The resampled image is then saved to the output path.

    Args:
        input_image_path (str): Path to the input .nii.gz image.
        output_image_path (str): Path where the resampled image will be saved.
        new_size (tuple, optional): Desired output size as a tuple of (Depth, Height, Width).
                                    Defaults to (512, 512, 512).

    Returns:
        None
    """
    # Read the input image
    input_image = sitk.ReadImage(input_image_path)

    # Retrieve original size (number of voxels) and spacing (physical size per voxel)
    original_size = input_image.GetSize()          # e.g., (D, H, W)
    original_spacing = input_image.GetSpacing()    # e.g., (s_D, s_H, s_W)

    # Compute the new spacing to preserve the overall physical dimensions:
    # new_spacing = (original_size * original_spacing) / new_size
    new_spacing = [osz * osp / nsz for osz, osp, nsz in zip(original_size, original_spacing, new_size)]

    # Set up the resampling filter
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(input_image.GetDirection())
    resample.SetOutputOrigin(input_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetInterpolator(sitk.sitkBSpline)

    # Execute the resampling
    resampled_image = resample.Execute(input_image)

    # Save the resampled image
    sitk.WriteImage(resampled_image, output_image_path)

    print(f"Resampled image saved to {output_image_path}")






