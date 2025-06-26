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
from .common import TrainingState, TrainerOverrides, ModelTrainerBase, ModelBase, TrainingConfig, BatchResults, ValidationResults, select_device
from .composite_dataset import CompositeDataset, sequence_collate_fn, BesCombine
from .models import SingleDigitModel, DigitSequenceModel

# class DavidCombine(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         train: bool,
#         output_width: int = 128,
#         output_height: int = 128,
#         output_batch_size: int = 1,
#         batches_per_epoch: int = 10,
#         line_height_min: int = 16,
#         line_height_max: int = 64,
#         line_spacing_min: int = -2,
#         line_spacing_max: int = 12,
#         horizontal_padding_min: int = -2,
#         horizontal_padding_max: int = 128,
#         first_line_offset: int = 2,
#         image_scaling_min: float = 0.7,
#         padding_token_id: int = -1,
#         start_token_id: int = 10,
#         end_token_id: int = 10,
#         max_sequence_length: int = 32,
#     ):
#         self.train = train
#         self.output_width = output_width
#         self.output_height = output_height
#         self.output_batch_size = output_batch_size
#         self.batches_per_epoch = batches_per_epoch
#         self.line_height_min = line_height_min
#         self.line_height_max = line_height_max
#         self.line_spacing_min = line_spacing_min
#         self.line_spacing_max = line_spacing_max
#         self.horizontal_padding_min = horizontal_padding_min
#         self.horizontal_padding_max = horizontal_padding_max
#         self.first_line_offset = first_line_offset
#         self.image_scaling_min = image_scaling_min
#         self.padding_token_id = padding_token_id
#         self.start_token_id = start_token_id
#         self.end_token_id = end_token_id
#         self.max_sequence_length = max_sequence_length
#
#     def __len__(self):
#         return self.output_batch_size * self.batches_per_epoch
#
#     def __getitem__(self, idx):

class IterableWithLength:
    def __init__(self, iterable, length):
        self.iterable = iterable
        self.length = length

    def __iter__(self):
        return iter(self.iterable)

    def __len__(self):
        return self.length

def composite_image_generator_david(
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
    left_margin_offset: int = 2,
    image_scaling_min: float = 0.7,
    padding_token_id: int = -1,
    start_token_id: int = 10,
    end_token_id: int = 10,
    max_sequence_length: int = 32,
):
    image_loader = DataLoader(
        image_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=2,
        pin_memory=select_device() == "cuda",  # Speed up CUDA
    )

    def loop_image_forever(batched_image_iterator):
        while True:
            for raw_batch in batched_image_iterator:
                images, labels = raw_batch
                for image, label in zip(images, labels):
                    yield image, label
            
    infinite_labelled_images = loop_image_forever(image_loader).__iter__()

    pre_generated = torch.rand((10000,))
    pre_generated_index = 0
    
    def rand():
        nonlocal pre_generated, pre_generated_index
        if pre_generated_index >= len(pre_generated):
            pre_generated = torch.rand((10000,))
            pre_generated_index = 0
        value = pre_generated[pre_generated_index]
        pre_generated_index += 1
        return value.item()

    def randint(min_value, max_value_inclusive):
        return math.floor(rand() * (max_value_inclusive + 1 - min_value)) + min_value

    def rand_biaseddown():
        return (math.exp(rand() * 2) - 1)/(math.exp(2) - 1)
    
    def randint_biaseddown(min_value, max_value_inclusive):
        return math.floor(rand_biaseddown() * (max_value_inclusive + 1 - min_value)) + min_value
    
    for batch_index in range(batches_per_epoch):
        composite_images = torch.zeros((output_batch_size, 1, output_height, output_width), dtype=torch.float32)
        in_labels = torch.zeros((output_batch_size, max_sequence_length), dtype=torch.int64)
        out_labels = torch.zeros((output_batch_size, max_sequence_length), dtype=torch.int64)

        for composite_in_batch_index in range(output_batch_size):
            line_offset = first_line_offset
            image_in_composite_index = 0
            allow_more_images = True

            in_labels[composite_in_batch_index, 0] = start_token_id

            # Generate lines
            while allow_more_images:
                line_spacing = randint_biaseddown(line_spacing_min, line_spacing_max)
                line_offset += line_spacing
                line_height = randint_biaseddown(line_height_min, line_height_max)

                line_end_offset = line_offset + line_height
                if line_end_offset > output_height:
                    break

                horizontal_offset = left_margin_offset
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
                    ] += image
                    in_labels[composite_in_batch_index, image_in_composite_index + 1] = label
                    out_labels[composite_in_batch_index, image_in_composite_index] = label
                    image_in_composite_index += 1

                    horizontal_offset += image_size

                    if image_in_composite_index == max_sequence_length - 1:
                        allow_more_images = False

                line_offset += line_height

            out_labels[composite_in_batch_index, image_in_composite_index] = end_token_id

            image_in_composite_index += 1

            while image_in_composite_index < max_sequence_length:
                in_labels[composite_in_batch_index, image_in_composite_index] = padding_token_id
                out_labels[composite_in_batch_index, image_in_composite_index] = padding_token_id
                image_in_composite_index += 1
                
        yield composite_images, in_labels, out_labels

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
    composite_batches = composite_image_generator_david(
        device='cpu',
        image_dataset=train_set,
    )
    
    for batch_index, (batch_images, batch_labels) in enumerate(composite_batches):
        for image, labels in zip(batch_images, batch_labels):
            print(f"Labels: {labels.tolist()}")
            shape = (image.shape[-2], image.shape[-1])
            plt.imshow(image.reshape(shape), cmap="gray")
            plt.show()
