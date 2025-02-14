# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset
import numpy as np

class Custom3DDataset(Dataset):
    """
    A custom dataset for loading 3D volumes stored in .npz files.

    Each .npz file should contain a key 'volume' with the corresponding numpy array.
    Optionally applies a transformation to the loaded volume.
    """
    def __init__(self, file_list_path: str, transform: Optional[callable] = None) -> None:
        """
        Args:
            file_list_path (str): Path to a text file listing .npz file paths.
            transform (callable, optional): A function/transform to apply to the volume.
        """
        self.transform = transform
        self.samples: List[str] = self._load_samples(file_list_path)

    def _load_samples(self, file_list_path: str) -> List[str]:
        with open(file_list_path, 'r') as file:
            files = [line.strip() for line in file if line.strip().endswith('.npz')]
        return files

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path = self.samples[idx]
        volume = torch.tensor(
            np.load(file_path)['volume'], dtype=torch.float
        ).unsqueeze(0)  # Add channel dimension

        if self.transform:
            volume = self.transform(volume)

        # Returning an arbitrary label; update as necessary.
        return volume, torch.tensor(0)


