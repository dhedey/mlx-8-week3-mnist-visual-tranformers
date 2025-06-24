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
from .common import TrainingState, TrainerParameters, ModelTrainerBase, ModelBase
from .composite_dataset import CompositeDataset, sequence_collate_fn

class EncoderOnlyModelTrainer(ModelTrainerBase):
    def __init__(
            self,
            model: ModelBase,
            parameters: TrainerParameters,
            continuation: Optional[TrainingState] = None,
        ):
        super().__init__(model=model, parameters=parameters, continuation=continuation)

        print("Preparing datasets...")

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

    def process_batch(self, raw_batch):
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

        print(f"Validation complete: {num_samples} samples, {total_correct} correct, {average_loss:.3g} average loss")
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

class ImageSequenceTransformerTrainer(ModelTrainerBase):
    def __init__(
            self,
            model: ModelBase,
            parameters: TrainerParameters,
            continuation: Optional[TrainingState] = None,
        ):
        super().__init__(model=model, parameters=parameters, continuation=continuation)

        print("Preparing datasets...")

        standard_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True), # Scale to [0, 1]
        ])

        data_folder = os.path.join(os.path.dirname(__file__), "datasets")
        train_sub_image_set = torchvision.datasets.MNIST(
            data_folder,
            download=True,
            transform=standard_transform,
            train=True,
        )
        test_sub_image_set = torchvision.datasets.MNIST(
            data_folder,
            download=True,
            transform=standard_transform,
            train=False,
        )

        min_digits = 0
        max_digits = 10
        self.max_sequence_length = max_digits + 1 # +1 for start and stop tokens
        canvas_size = (256, 256)

        train_composite_dataset = CompositeDataset(
            dataset=train_sub_image_set,
            length=100000,
            min_digits=min_digits,
            max_digits=max_digits,
            canvas_size=canvas_size,
            digit_size=28,
        )
        test_composite_dataset = CompositeDataset(
            dataset=test_sub_image_set,
            length=10000,
            min_digits=min_digits,
            max_digits=max_digits,
            canvas_size=canvas_size,
            digit_size=28,
        )

        device = self.model.get_device()
        pin_memory = device == 'cuda'
        self.train_loader = torch.utils.data.DataLoader(
            train_composite_dataset,
            batch_size=model.training_parameters.batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=self.collate_fn,
            pin_memory=pin_memory, # Speed up CUDA
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_composite_dataset,
            batch_size=model.training_parameters.batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=self.collate_fn,
            pin_memory=pin_memory, # Speed up CUDA
        )
        print(f"Training set size: {len(train_composite_dataset)}")
        print(f"Test set size: {len(test_composite_dataset)}")

    def collate_fn(self, batch):
        return sequence_collate_fn(
            batch,
            max_seq_length=self.max_sequence_length,
            pad_token_id=-1,
        )

    def get_train_data_loader(self):
        return self.train_loader

    def process_batch(self, raw_batch):
        images, input_sequences, expected_sequences = raw_batch

        device = self.model.get_device()
        images = images.to(device)
        input_sequences = input_sequences.to(device)
        expected_sequences = expected_sequences.to(device) # Shape: (BatchSize, SequenceLength) => VocabularyIndex

        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        logits: torch.Tensor = self.model(images, input_sequences)   # Shape: (BatchSize, SequenceLength, VocabularySize=11) => Probability Source

        # CrossEntropyLoss apparently needs a shape like (BatchSize, SequenceLength, ...)
        loss = criterion(logits.transpose(-1, -2), expected_sequences)

        return {
            "total_loss": loss,
            "logits": logits,
            "num_samples": len(images),
        }
    
    def _validate(self):
        total_loss = 0.0

        totals_by_label = [0] * 11
        correct_by_label = [0] * 11

        total_subimages = 0
        total_subimages_correct = 0
        total_probability_of_subimage_correct = 0.0

        total_composites = 0
        correct_composites = 0

        for raw_batch in self.test_loader:
            batch_results = self.process_batch(raw_batch)
            logits = batch_results["logits"]
            expected_sequences = batch_results["expected_sequences"]
            loss = batch_results["loss"]
            probabilities = F.softmax(logits, dim=1)

            for instance_logits, instance_expected, instance_probabilities in zip(logits, expected_sequences, probabilities):
                all_correct = True
                for sub_image_logits, expected_label, sub_image_probabilities in zip(instance_logits, instance_expected, instance_probabilities):
                    if expected_label == -1:
                        continue
                    expected_label = expected_label.item()
                    predicted_label = sub_image_logits.argmax().item()
                    is_correct = predicted_label == expected_label
                    totals_by_label[expected_label] += 1
                    if is_correct:
                        total_subimages_correct += 1
                        correct_by_label[expected_label] += 1
                    else:
                        all_correct = False
                    total_probability_of_subimage_correct += instance_probabilities[expected_label].item()
                    total_subimages += 1

                total_composites += 1
                if all_correct:
                    correct_composites += 1
            
            total_loss += loss.item()

        proportion_subimages_correct = total_subimages_correct / total_subimages if total_subimages > 0 else 0.0
        average_probability_of_subimages_correct = total_probability_of_subimage_correct / total_subimages if total_subimages > 0 else 0.0
        average_loss_per_subimage = total_loss / total_subimages if total_subimages > 0 else 0.0

        proportion_composites_correct = correct_composites / total_composites if total_composites > 0 else 0.0

        print(f"Validation complete: {total_composites} composites, containing {total_subimages} subimages, {total_subimages_correct} correct, {average_loss_per_subimage:.3g} average loss per subimage")
        print(f"* Accuracy: {proportion_subimages_correct:.2%} subimages correct")
        print(f"* Accuracy: {proportion_composites_correct:.2%} composites fully correct")
        print(f"* Average loss for each subimage: {average_loss_per_subimage}")
        print(f"* Average confidence in each subimage: {average_probability_of_subimages_correct:.2%}")
        print()
        print("Proportion subimages correct by actual label:")
        for i in range(10):
            label_prop_correct = correct_by_label[i] / totals_by_label[i] if totals_by_label[i] > 0 else 0
            if i == 10:
                index_label = "END"
            else:
                index_label = f"[{i}]"
            print(f"* {index_label}: {label_prop_correct:.2%} ({correct_by_label[i]} of {totals_by_label[i]})")
        print()

        return {
            "average_loss": average_loss_per_subimage,
            "proportion_subimages_correct": proportion_subimages_correct,
            "proportion_composites_correct": proportion_composites_correct,
            "average_probability_of_subimages_correct": average_probability_of_subimages_correct,
            "totals_by_label": totals_by_label,
        }

def composite_image_generator(
        device: str,
        image_dataset: datasets.Dataset,
        output_width: int = 128,
        output_height: int = 128,
        output_batch_size: int = 1,
        batches_per_epoch: int = 10,
        line_height_min: int = 16,
        line_height_max: int = 64,
        line_spacing_min: int = -2,
        line_spacing_max: int = 12,
        horizontal_padding_min: int = -2,
        horizontal_padding_max: int = 128,
        first_line_offset: int = 2,
        image_scaling_min: float = 0.7,
        start_token_id = 10,
        end_token_id = 11,
        max_labels_per_image: int = 32,
    ):
    pin_memory = device == 'cuda'
    
    image_loader = DataLoader(
        image_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=2,
        pin_memory=pin_memory,  # Speed up CUDA
    )

    def loop_image_forever(batched_image_iterator):
        while True:
            for raw_batch in batched_image_iterator:
                images, labels = raw_batch
                images = images.to(device)
                labels = labels.to(device)
                for image, label in zip(images, labels):
                    yield image, label 
            
    infinite_labelled_images = loop_image_forever(image_loader).__iter__()

    rand_generator = torch.Generator(device="cpu")

    def randint(min_value, max_value_inclusive):
        nonlocal rand_generator
        return torch.randint(min_value, max_value_inclusive + 1, (1,), generator=rand_generator).item()
    
    def rand():
        nonlocal rand_generator
        return torch.rand((1,), generator=rand_generator).item()
    
    def rand_biaseddown():
        return (math.exp(rand() * 2) - 1)/(math.exp(2) - 1)
    
    def randint_biaseddown(min_value, max_value):
        return math.floor(rand_biaseddown() * (max_value + 1 - min_value)) + min_value
    
    for batch_index in range(batches_per_epoch):
        composite_images = torch.zeros((output_batch_size, 1, output_height, output_width), device=device)
        composite_labels = torch.zeros((output_batch_size, max_labels_per_image), dtype=torch.int64, device=device)

        for composite_in_batch_index in range(output_batch_size):
            line_offset = first_line_offset
            image_in_composite_index = 0
            allow_more_images = True

            # Generate lines
            while allow_more_images:
                line_spacing = randint_biaseddown(line_spacing_min, line_spacing_max)
                line_offset += line_spacing
                line_height = randint_biaseddown(line_height_min, line_height_max)

                line_end_offset = line_offset + line_height
                if line_end_offset > output_height:
                    break

                horizontal_offset = 0
                while allow_more_images:
                    horizontal_padding = randint_biaseddown(horizontal_padding_min, horizontal_padding_max)
                    horizontal_offset += horizontal_padding

                    # Fill the line with random images
                    image_size = randint_biaseddown(math.floor(image_scaling_min * line_height), line_height)

                    horizontal_end_offset = horizontal_offset + image_size
                    if horizontal_end_offset > output_width:
                        break

                    image, label = next(infinite_labelled_images)
                     # NB: image is already rotated!
                    image = v2.Resize(image_size)(image)

                    image_vertical_start_offset = line_offset + randint(0, line_height - image_size)
                    image_vertical_end_offset = image_vertical_start_offset + image_size

                    composite_images[
                        composite_in_batch_index,
                        0:1,
                        image_vertical_start_offset:image_vertical_end_offset,
                        horizontal_offset:horizontal_end_offset
                    ] += image.to(device)
                    composite_labels[composite_in_batch_index, image_in_composite_index] = label
                    image_in_composite_index += 1

                    horizontal_offset += image_size

                    if image_in_composite_index == max_labels_per_image - 1:
                        allow_more_images = False

                line_offset += line_height
            composite_labels[composite_in_batch_index, image_in_composite_index] = end_token_id
            
        yield composite_images, composite_labels

if __name__ == "__main__":
    training_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True), # Scale to [0, 1]
        v2.RandomResize(28, 40),
        v2.RandomRotation(25),
        v2.RandomResizedCrop(size = 28, scale = (28.0/40, 28.0/40)),
    ])

    data_folder = os.path.join(os.path.dirname(__file__), "datasets")
    train_set = torchvision.datasets.MNIST(
        data_folder,
        download=True,
        transform=training_transform,
        train=True,
    )
    composite_batches = composite_image_generator(
        device='cpu',
        image_dataset=train_set,
    )
    
    for batch_index, (batch_images, batch_labels) in enumerate(composite_batches):
        for image, labels in zip(batch_images, batch_labels):
            print(f"Labels: {labels.tolist()}")
            shape = (image.shape[-2], image.shape[-1])
            plt.imshow(image.reshape(shape), cmap="gray")
            plt.show()
