import gemmi
import numpy as np
import sys
from typing import Tuple, Union

"""
Script to convert a Patterson Map in .map.gz format
to an interpolated 3D numpy array on a 128 A^3 carterian grid
"""

def get_p_map_grid(file_path: str) -> np.array:
        return gemmi.read_ccp4_map(file_path).grid

def get_p_map_corrected(
        file_path: str,
        grid_size: Tuple[float, float, float],
        grid_spacing: float) -> Union[np.array, None]:

    grid = get_p_map_grid(file_path)
    arr = np.zeros(grid_size, dtype=np.float32)

    tr = gemmi.Transform()
    tr.mat.fromlist([[grid_spacing, 0, 0], [0, grid_spacing, 0], [0, 0, grid_spacing]])
    tr.vec.fromlist([-grid_size[0]/2., -grid_size[1]/2., -grid_size[2]/2.])
    grid.interpolate_values(arr, tr)
    return arr

if __name__ == "__main__":

    txt_file = sys.argv[1]
    grid_size = (256, 256, 256)
    grid_spacing = 0.5

    with open(txt_file, "r") as g:
        lines = g.readlines()

    for line in lines:
        file_path = line.split()[-1]
        output_path = file_path.replace(".map.gz", ".npy")
        arr = get_p_map_corrected(file_path=file_path, grid_size=grid_size, grid_spacing=grid_spacing)
        np.save(output_path, arr)

