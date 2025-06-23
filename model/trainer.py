from dataclasses import dataclass, field
import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
import statistics
import random
import wandb
from typing import Optional
import time
from .common import TrainingState, ModelTrainerBase
from .models import ModelBase

class ModelTrainer(ModelTrainerBase):
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

        # TODO: Initialize datasets and data loaders here

    def process_test_batch(self):
        # Implement the logic to process a batch of data for training
        # This should return a dictionary with the total loss and number of samples processed
        raise NotImplementedError("To be implemented.")
        return {
            "total_loss": torch.tensor(0.0),  # Placeholder
            "num_samples": 0,  # Placeholder
        }
    
    def validate(self):
        print()
        print("== VALIDATING MODEL ==")
        print()
        
        raise NotImplementedError("To be implemented.")

        return {
            # TODO: Add validation metrics
        }
