
import sys
import onnxruntime as ort
import numpy as np
import os
from pathlib import Path
import shlex, subprocess
import time
import gemmi
from typing import Generator

"""
Author: David McDonagh (david.mcdonagh@stfc.ac.uk)

Script for calculating protein crystal solvent content using a modified  
ViT-V-Net transformer model.

ViT-V-Net paper: https://arxiv.org/abs/2104.06468,

or a modified UNet3D model based on this work: 

https://doi.org/10.7554/eLife.57613

Requirements:

    numpy
    ccp4 (ecalc, fft, gemmi, cif2mtz)
    onnxruntime (pip install onnxruntime) or onnxruntime-gpu (pip install onnxruntime-gpu)

Usage:

    python predict_solvent_content.py example.map
    python predict_solvent_content.py example.mtz
    python predict_solvent_content.py example.cif
    python predict_solvent_content.py example.txt

    where example.txt contains a list of .map, .mtz, or .cif files
    
    Output will be the name of each file followed by the solvent content (%)

    Default model is unet_model.onnx, but this can be overridden by specifiying the model
    as the second parameter. E.g.

    python predict_solvent_content.py example.map vitvnet_model.onnx

Notes:

    .map files are assumed to be normalised 
    e.g. generated from CCP4 from a .mtz file using 

    ecalc hklin {mtz_file_path} hklout {ecalc_mtz_file_path}", 
        keywords=f"labin FP={F} SIGFP={SIGF}\nlabout FECALC=DISO E=E F2OR=F2OR E2OR=E2OR\nend\n", 
    )

    fft hklin {ecalc_mtz_file_path} mapout {ecalc_map_file_path}", 
        keywords="labin F1=E\npatterson\nend\n"

    By default the onnx model is run on the cpu.
    This can be changed to the gpu by setting the providers to 
    
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]

"""


def parse_input(file_path: str, grid_size: int, grid_spacing: float) -> Generator[str, np.array, None]:

    """
    file_path: accepts:
                - .txt file containing file paths to .mtz, .map, or .cif files
                - .mtz file
                - .cif file
                - .map file
    grid_size: the size of the map calculated for all structures
    grid_spacing: spacing between points in the map (Angstroms)

    returns a generator of Patterson maps as numpy arrays
    """

    def run(command_line: str, keywords:str ="") -> None:

        """ Wrapper to run a subprocess command """

        # Launch command
        if os.name == "nt":
            process_args = shlex.split(command_line, posix=False)
            p = subprocess.Popen(process_args, shell="True", stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        else:
            process_args = shlex.split(command_line)
            p = subprocess.Popen(process_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

        (outp, inp) = (p.stdout, p.stdin)

        if keywords != "":
            inp.write(keywords.encode())

        inp.close()
        line = outp.readline().decode()
        while line:
            line = outp.readline().decode()
        outp.close()


    def parse_patterson_map(map_file_path: str, grid_size: int, grid_spacing: float) -> np.array:
        raw_map = gemmi.read_ccp4_map(map_file_path).grid
        arr = np.zeros(grid_size, dtype=np.float32)
        tr = gemmi.Transform()
        tr.mat.fromlist([[grid_spacing, 0, 0], [0, grid_spacing, 0], [0, 0, grid_spacing]])
        tr.vec.fromlist([-grid_size[0]/2., -grid_size[1]/2., -grid_size[2]/2.])
        raw_map.interpolate_values(arr, tr)
        return arr.reshape(1, 1, grid_size[0], grid_size[1], grid_size[2])

    def parse_patterson_map_from_mtz(mtz_file_path: str, grid_size: int, grid_spacing: float) -> np.array:

        mtz = gemmi.read_mtz_file(mtz_file_path)
        F=mtz.columns_with_type("F")[0].label
        SIGF=mtz.columns_with_type("Q")[0].label
        FREE=mtz.columns_with_type("I")[0].label

        mtz_file_path = Path(mtz_file_path)
        ecalc_mtz_file_path = mtz_file_path.with_name(f"ecalc_{mtz_file_path.stem}_E.mtz")
        ecalc_map_file_path = mtz_file_path.with_name(f"ecalc_{mtz_file_path.stem}_E.map")

        run(
            f"ecalc hklin {mtz_file_path} hklout {ecalc_mtz_file_path}", 
            keywords=f"labin FP={F} SIGFP={SIGF}\nlabout FECALC=DISO E=E F2OR=F2OR E2OR=E2OR\nend\n", 
        )
        run(
            f"fft hklin {ecalc_mtz_file_path} mapout {ecalc_map_file_path}", 
            keywords="labin F1=E\npatterson\nend\n"
        )

        pmap = parse_patterson_map(
                map_file_path=ecalc_map_file_path._str, 
                grid_size=grid_size,
                grid_spacing=grid_spacing
            )
        
        os.remove(ecalc_mtz_file_path)
        os.remove(ecalc_map_file_path)
        
        return pmap

    def parse_patterson_map_from_cif(cif_file_path: str, grid_size: int, grid_spacing: float) -> np.array:

        def get_crystal_info(cif_file_path):
            with open(cif_file_path, "r") as g:
                lines = g.readlines()
            crystal = "0.0 0.0 0.0 0.0 0.0 0.0"
            space_group = None

            crystal_lst = []
            cell_lines = [
                    "_cell.length_a", 
                    "_cell.length_b",
                    "_cell.length_c",
                    "_cell.angle_alpha",
                    "_cell.angle_beta",
                    "_cell.angle_gamma",
                    ]
            for line in lines:
                ls = line.split()
                if ls[0] in cell_lines:
                    crystal_lst.append(ls[-1])
                elif line.startswith("_symmetry.space_group_name_H-M"):
                    space_group = line.split("_symmetry.space_group_name_H-M")[-1]
            if len(crystal_lst) == 6:
                crystal = " ".join(crystal_lst)

            return crystal, space_group


        cif_file_path = Path(cif_file_path)
        crystal, space_group = get_crystal_info(cif_file_path)
        mtz_file_path = cif_file_path.with_name(f"{cif_file_path.stem}.mtz")

        run(
            f"cif2mtz hklin {cif_file_path} hklout {mtz_file_path}",
            keywords=f"cell {crystal}\nsymm {space_group}\nend"
        )

        pmap = parse_patterson_map_from_mtz(
                mtz_file_path=mtz_file_path._str,
                grid_size=grid_size,
                grid_spacing=grid_spacing
            )

        os.remove(mtz_file_path)

        return pmap


    extension = Path(file_path).suffix
    if extension == ".txt":
        with open(file_path, "r") as g:
            filenames = [i.rstrip() for i in g.readlines()]
    else:
        filenames = [file_path]

    input_files = {}
    for filename in filenames:
        p_filename = Path(filename)
        name = p_filename.name
        extension = p_filename.suffixes[-1]
        if extension == ".map":
            yield name, parse_patterson_map(filename, grid_size, grid_spacing)
        elif extension == ".mtz":
            yield name, parse_patterson_map_from_mtz(filename, grid_size, grid_spacing)
        elif extension == ".cif":
            yield name, parse_patterson_map_from_cif(filename, grid_size, grid_spacing)


if __name__ ==  "__main__":

    # params used when training vitvnet_model.onnx and unet_model.onnx
    grid_size = (256, 256, 256)
    grid_spacing = .5

    # cpu by default, for gpu use providers = ["CUDAExecutionProvider"]
    providers = ["CPUExecutionProvider"]

    # Load input
    input_data = parse_input(sys.argv[1], grid_size=grid_size, grid_spacing=grid_spacing)

    # Load model
    if len(sys.argv) == 3:
        onnx_model_path = os.path.join(os.getcwd(), sys.argv[2])
    else: # Default to unet
        onnx_model_path = os.path.join(os.getcwd(), "unet_model.onnx")
    onnx_model = ort.InferenceSession(
                onnx_model_path,
                providers=providers
              )
    for name, pmap in input_data:
        result = onnx_model.run(None, {"input": pmap})
        print(f"{name}: {round(result[0][0][0]*100, 3)}")

