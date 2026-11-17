from itertools import permutations
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset

from dl_common import NUM_DL_METADATA

_AXIS_PERMS: Tuple[Tuple[int, int, int], ...] = tuple(permutations((0, 1, 2)))


def augment_patterson_3d(arr: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Apply a random element of the 48-element 3D octahedral cube symmetry group O_h
    (6 axis permutations x 8 sign reflections) while keeping the Patterson origin
    strictly centered at `(N_x // 2, N_y // 2, N_z // 2)`.
    """
    if rng is None:
        perm = _AXIS_PERMS[np.random.randint(0, 6)]
        flips = np.random.randint(0, 2, size=3)
    else:
        perm = _AXIS_PERMS[int(rng.integers(0, 6))]
        flips = rng.integers(0, 2, size=3)

    out = np.transpose(arr, perm)
    for axis in range(3):
        if flips[axis]:
            out = np.flip(out, axis=axis)
            # On an even grid x_i = (i - N/2)*dx, reversing maps N/2 -> N/2 - 1;
            # rolling by +1 restores exact fixed-point symmetry x -> -x around N/2.
            if out.shape[axis] % 2 == 0:
                out = np.roll(out, shift=1, axis=axis)
    return np.ascontiguousarray(out)


class MapDataset(Dataset):
    """
    Dataset for loading 3D Patterson map .npy files (float16 or float32),
    optional 3D cube-group augmentations, pre-phasing metadata vectors, and
    inverse label-density sample weights.
    """

    def __init__(
        self,
        file_paths: Sequence[str],
        solvent_content: Union[Sequence[float], np.ndarray, torch.Tensor],
        names: List[str],
        metadata_map: Optional[Dict[str, np.ndarray]] = None,
        sample_weights: Optional[Union[Sequence[float], np.ndarray, torch.Tensor]] = None,
        augment: bool = False,
        return_extra: bool = False,
    ) -> None:
        assert len(solvent_content) == len(file_paths) == len(names)
        self.file_paths = list(file_paths)
        self.solvent_content = torch.as_tensor(solvent_content, dtype=torch.float32)
        self.names = list(names)
        self.augment = augment
        self.return_extra = return_extra

        nan_vec = np.full(NUM_DL_METADATA, np.nan, dtype=np.float32)
        meta_rows = []
        for nm in self.names:
            key = nm.split(".")[0].lower()
            vec = metadata_map.get(key) if metadata_map is not None else None
            meta_rows.append(vec if vec is not None else nan_vec)
        self.metadata = torch.as_tensor(np.stack(meta_rows, axis=0), dtype=torch.float32)

        if sample_weights is not None:
            self.sample_weights = torch.as_tensor(sample_weights, dtype=torch.float32).view(-1, 1)
        else:
            self.sample_weights = torch.ones((len(self.names), 1), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        p_map = np.load(self.file_paths[idx]).astype(np.float32, copy=False)
        if self.augment:
            p_map = augment_patterson_3d(p_map)
        vol = np.expand_dims(p_map, axis=0)
        sc = self.solvent_content[idx]
        name = self.names[idx]
        if self.return_extra:
            return vol, self.metadata[idx], sc, self.sample_weights[idx], name
        return vol, sc, name
