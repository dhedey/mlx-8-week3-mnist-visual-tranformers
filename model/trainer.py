from dataclasses import dataclass, field
import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision
from torchvision.transforms import v2
import statistics
import random
import wandb
from typing import Optional
import time
from .common import TrainingState, ModelTrainerBase
from .models import ModelBase

class EncoderOnlyModelTrainer(ModelTrainerBase):
    def __init__(
            self,
            model: ModelBase,
            continuation: Optional[TrainingState] = None,
            override_to_epoch: Optional[int] = None,
            validate_after_epochs: int = 5,
            immediate_validation: bool = False,
        ):
        super().__init__(
            model=model,
            continuation=continuation,
            override_to_epoch=override_to_epoch,
            validate_after_epochs=validate_after_epochs,
            immediate_validation=immediate_validation
        )

        training_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True), # Scale to [0, 1]
            v2.RandomResize(28, 40),
            v2.RandomRotation(30),
            v2.RandomResizedCrop(size = 28, scale = (28.0/40, 28.0/40)),
        ])

        test_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True), # Scale to [0, 1]
        ])

        train_set = torchvision.datasets.MNIST(
            "./data",
            download=True,
            transform=training_transform,
            train=True,
        )
        test_set = torchvision.datasets.MNIST(
            "./data",
            download=True,
            transform=test_transform,
            train=False,
        )
        print(f"Training set size: {len(train_set)}")
        print(f"Test set size: {len(test_set)}")

    def process_test_batch(self):
        # Implement the logic to process a batch of data for training
        # This should return a dictionary with the total loss and number of samples processed
        raise NotImplementedError("To be implemented.")
        return {
            "total_loss": torch.tensor(0.0),  # Placeholder
            "num_samples": 0,  # Placeholder
        }
    
    def validate(self):
        raise NotImplementedError("To be implemented.")

        return {
            # TODO: Add validation metrics
        }
