from typing import List, Sequence, Union
import numpy as np
import torch
from torch.utils.data import Dataset


class MapDataset(Dataset):
    """Dataset for loading 3D Patterson map .npy files (float16 or float32)."""

    def __init__(
        self,
        file_paths: Sequence[str],
        solvent_content: Union[Sequence[float], np.ndarray, torch.Tensor],
        names: List[str],
    ) -> None:
        assert len(solvent_content) == len(file_paths) == len(names)
        self.file_paths = list(file_paths)
        self.solvent_content = torch.as_tensor(solvent_content, dtype=torch.float32)
        self.names = list(names)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        p_map = np.load(self.file_paths[idx]).astype(np.float32, copy=False)
        sc = self.solvent_content[idx]
        name = self.names[idx]
        return np.expand_dims(p_map, axis=0), sc, name
