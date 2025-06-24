from dataclasses import dataclass, field
import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import statistics
import random
import wandb
import math
import os
from typing import Optional
import time
from .common import TrainingState, TrainerParameters, ModelTrainerBase, ModelBase, PersistableModel

for model_name in [
    "encoder-only",
    "encoder-only-best",
    "encoder-only-big",
    "encoder-only-big-best",
    "encoder-only-bigger",
    "encoder-only-bigger-best",
    "encoder-only-positional",
    "encoder-only-positional-best",
    "encoder-only-positional-dropout",
    "encoder-only-positional-dropout-best",
]:
    path = PersistableModel._model_path(model_name)
    data = torch.load(path)
    # if data["model"]["creation_state"]["hyper_parameters"]["heads_per_layer"] > 1:
    #     print(f"Fixing {model_name} attention heads from {data['model']['creation_state']["hyper_parameters"]['heads_per_layer']} to 1")
    #     data["model"]["creation_state"]["hyper_parameters"]["heads_per_layer"] = 1
    #     torch.save(data, path)
    if "model_trainer_class_name" not in data["training"]:
        data["training"]["model_trainer_class_name"] = "EncoderOnlyModelTrainer"
        torch.save(data, path)