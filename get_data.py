from typing import List
import numpy as np
from torch.utils.data import Dataset

class MapDataset(Dataset):

    """
    Class for loading Patterson Maps
    """

    def __init__(self, 
            file_paths: List[str], 
            solvent_content: List[str], 
            names: List[str]
            ) -> None:

        assert len(solvent_content) == len(file_paths) == len(names)
        self.file_paths = file_paths
        self.solvent_content=solvent_content
        self.names=names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        p_map = np.load(self.file_paths[idx])
        sc = self.solvent_content[idx]
        name = self.names[idx]
        return np.expand_dims(p_map, axis=0), sc, name
