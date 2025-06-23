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
import os
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

        data_folder = os.path.join(os.path.dirname(__file__), "datasets")
        train_set = torchvision.datasets.MNIST(
            data_folder,
            download=True,
            transform=training_transform,
            train=True,
        )
        test_set = torchvision.datasets.MNIST(
            data_folder,
            download=True,
            transform=test_transform,
            train=False,
        )
        device = self.model.get_device()
        pin_memory = device == 'cuda'
        self.train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=model.training_parameters.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=pin_memory, # Speed up CUDA
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=model.training_parameters.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=pin_memory, # Speed up CUDA
        )
        print(f"Training set size: {len(train_set)}")
        print(f"Test set size: {len(test_set)}")

    def get_train_data_loader(self):
        return self.train_loader

    def process_test_batch(self, raw_batch):
        inputs, labels = raw_batch

        criterion = nn.CrossEntropyLoss()
        logits = self.model(inputs)
        loss = criterion(logits, labels.to(self.model.get_device()))

        return {
            "total_loss": loss,
            "num_samples": len(inputs),
        }
    
    def _validate(self):
        total_loss = 0.0
        num_samples = 0

        totals_by_label = [0] * 10
        correct_by_label = [0] * 10

        total_correct = 0
        total_probability_of_correct = 0.0

        with torch.no_grad():
            self.model.eval()
            for raw_batch in self.test_loader:
                inputs, labels = raw_batch # Shape: (BatchSize, Channels=1, 28, 28) and (BatchSize)

                criterion = nn.CrossEntropyLoss()
                logits = self.model(inputs) # Shape: (BatchSize, 10)
                loss = criterion(logits, labels.to(logits.device))
                probabilities = F.softmax(logits, dim=1)

                for instance_logits, instance_label, instance_probabilities in zip(logits, labels, probabilities):
                    instance_label = instance_label.item()
                    predicted_label = instance_logits.argmax().item()
                    is_correct = predicted_label == instance_label
                    totals_by_label[instance_label] += 1
                    if is_correct:
                        total_correct += 1
                        correct_by_label[instance_label] += 1
                    total_probability_of_correct += instance_probabilities[instance_label].item()

                total_loss += loss.item()
                num_samples += len(inputs)

        proportion_correct = total_correct / num_samples if num_samples > 0 else 0.0
        average_probability_of_correct = total_probability_of_correct / num_samples if num_samples > 0 else 0.0
        average_loss = total_loss / num_samples if num_samples > 0 else 0.0

        print(f"Validation complete: {num_samples} samples, {total_correct} correct, {average_loss:.2} average loss")
        print(f"* Accuracy: {proportion_correct:.2%} correct")
        print(f"* Average Loss: {average_loss}")
        print(f"* Average confidence in correct answer: {average_probability_of_correct:.2%}")
        print()
        print("Proportion correct by actual label:")
        for i in range(10):
            label_prop_correct = correct_by_label[i] / totals_by_label[i] if totals_by_label[i] > 0 else 0
            print(f"* {i}: {label_prop_correct:.2%} ({correct_by_label[i]} of {totals_by_label[i]})")
        print()

        return {
            "average_loss": average_loss,
            "proportion_correct": proportion_correct,
            "average_probability_of_correct": average_probability_of_correct,
            "totals_by_label": totals_by_label,
        }
