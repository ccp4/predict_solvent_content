
import sys
from get_data import MapDataset
from usolv import ResidualUNetSE3D
from vitsolv import ViTVNet, get_3DReg_config
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from random import shuffle
import time
from typing import Tuple, List

"""
Script for training the unet and vitvnet models

Usage

python train.py training_data.txt validation_data.txt <model>

where model can be vitvet or unet
"""

class WeightedMSE(nn.Module):

    """
    Custom MSE cost function to weight more heavily the structures 
    with high/low solvent content
    """

    def __init__(self, weight):
        super(WeightedMSE, self).__init__()
        self.weight = weight

    def forward(self, predictions, targets):
        loss = torch.mean((predictions - targets)**2 * (abs(targets - 0.5)*self.weight + 1))
        return loss

def get_training_data(training_input_file_path: str, validation_input_file_path: str) -> Tuple[List, np.ndarray, List, List, np.ndarray, List]:

    with open(training_input_file_path, "r") as g:
        training_data = g.readlines()

    shuffle(training_data)
    training_input = []
    training_output = []
    training_names = []

    for i in training_data:
        ls = i.split()
        training_input.append(ls[2].rstrip())
        training_output.append(ls[1])
        training_names.append(ls[0])
    training_output = np.expand_dims(np.array([float(i.rstrip())/100. for i in training_output]), axis=1)

    with open(validation_input_file_path, "r") as g:
        validation_data = g.readlines()

    validation_input = []
    validation_output = []
    validation_names = []

    for i in validation_data:
        ls = i.split()
        validation_input.append(ls[2].rstrip())
        validation_output.append(ls[1])
        validation_names.append(ls[0])
    validation_output = np.expand_dims(np.array([float(i.rstrip())/100. for i in validation_output]), axis=1)
    return training_input, training_output, training_names, validation_input, validation_output, validation_names

if __name__ ==  "__main__":

    abs_path = os.getcwd()
    training_input_file_path = os.path.join(abs_path, sys.argv[1])
    validation_input_file_path = os.path.join(abs_path, sys.argv[2])

    model_type = sys.argv[3]
    if model_type == "vitvnet":
        config = get_3DReg_config()
        model = ViTVNet(config=config)

    elif model_type == "unet":
        model = ResidualUNetSE3D(
                in_channels=1,
                f_maps=[16, 32, 64, 128, 256]

                )

    batch_size = 6
    epochs = 40
    loss_weight = 8

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

    torch.cuda.empty_cache()

    training_input, training_output, training_names, validation_input, validation_output, validation_names = get_training_data(training_input_file_path, validation_input_file_path)

    training_data = MapDataset(training_input, torch.Tensor(training_output), training_names)
    validation_data = MapDataset(validation_input, validation_output, validation_names)
    print(f"Training data size: {len(training_input)}")
    print(f"Validation data size: {len(validation_input)}")

    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)

    model = nn.DataParallel(model)
    model.cuda()
    loss_fn = WeightedMSE(loss_weight)

    lr=0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    epoch_start=0

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        epoch_start = int(sys.argv[2])
        print(f"Loading model {model_path}")
        model.load_state_dict(torch.load(model_path))


    for epoch in range(epoch_start, epochs):

        print(f"{epoch=}")
        training_file = open(os.path.join(abs_path, f"training_correlation_{epoch}.txt"), "w")
        batch_start_time = time.time()
        for i, data in enumerate(training_loader):
            inputs, labels, names = data
            inputs = inputs.to(torch.float32)
            inputs = inputs.cuda()
            labels = labels.to(torch.float32)
            labels = labels.cuda()
            optimizer.zero_grad()
            
            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()
            for count, j in enumerate(labels):
                print(f"{names[count]} {j.tolist()[0]} {outputs[count].tolist()[0]}")
                training_file.write(f"{names[count]} {j.tolist()[0]} {outputs[count].tolist()[0]}\n")

        training_file.close()
        model_path = f"{abs_path}/models/model_{epoch}"
        print(f"Saving model to {model_path}")
        torch.save(model.state_dict(), model_path)

        if epoch <= 2:
            continue

        print("validation")
        validation_file = open(os.path.join(abs_path, f"validation_correlation_{epoch}.txt"), "w")
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels, vnames = vdata
                vinputs = vinputs.to(torch.float32)
                vinputs = vinputs.cuda()
                vlabels = vlabels.to(torch.float32)
                vlabels = vlabels.cuda()
                voutputs = model(vinputs)
                for count, j in enumerate(vlabels):
                    print(f"{vnames[count]} {j.tolist()[0]} {voutputs[count].tolist()[0]}")
                    validation_file.write(f"{vnames[count]} {j.tolist()[0]} {voutputs[count].tolist()[0]}\n")

        validation_file.close()
