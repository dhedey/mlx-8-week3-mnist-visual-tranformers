from .create_composite import get_composite_image_and_sequence, load_mnist_dataset
from torch.utils.data import Dataset
import torch
import torchvision
import numpy as np
import os
import einops
from tqdm import tqdm

class BesCombine(Dataset):
  def __init__(self, train=True, h_patches = 2, w_patches = 2, length = None, p_skip = 0):
    super().__init__()
    self.h_patches = h_patches
    self.w_patches = w_patches
    self.p_skip = p_skip

    self.tk = { '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 's': 10}
    gen = np.random.default_rng(42)
    data_folder = os.path.join(os.path.dirname(__file__), "datasets")
    ds = torchvision.datasets.MNIST(root=data_folder, train=train, download=True)
    self.ln = len(ds) if length is None else length

    # Pre-process all images and cache them in memory; first image is 0
    all_images = torch.cat((torch.zeros(1, 1, 28, 28), ds.data.unsqueeze(1).float() / 255.0))
    normalizer = torchvision.transforms.Normalize((0.1307,), (0.3081,))
    self.processed_images = normalizer(all_images)
    self.all_labels = torch.cat((torch.zeros(1), ds.targets)).to(torch.long)
    # Create a deterministic map of indices, so __getitem__(i) is always the same 
    self.index_map = gen.choice(range(1, len(ds) + 1), size = (self.ln, self.h_patches * self.w_patches), replace=True)
    self.skip_map = gen.binomial(1, self.p_skip, size=(self.ln, self.h_patches * self.w_patches))

  def __len__(self):
    return self.ln

  def __getitem__(self, idx):
    # Get the pre-determined list of 4 indices for this item
    image_indices = self.index_map[idx]
    skip_indices = self.skip_map[idx]
    image_indices = np.logical_not(skip_indices) * image_indices
    # Retrieve the pre-processed images and labels using fast tensor indexing
    stack = self.processed_images[image_indices].squeeze(1) # shape: (h_patches * w_patches, 28, 28)
    #only retrieve labels for non-skipped indices
    label = self.all_labels[image_indices[np.logical_not(skip_indices)]]

    combo = einops.rearrange(stack, '(h w) ph pw -> (h ph) (w pw)', h=self.h_patches, w=self.w_patches, ph=28, pw=28)
    #patch = einops.rearrange(combo, '(h ph) (w pw) -> (h w) ph pw', ph=14, pw=14)
    #label = [10] + label + [11]
    return combo, label


class CompositeDataset(Dataset):
    def __init__(self, dataset = None, length = 100000, min_digits = 1, max_digits = 5, canvas_size=(256, 256), digit_size=28):
        if dataset is None:
            self.dataset = load_mnist_dataset()
        else:
            self.dataset = dataset
        self.length = length
        self.min_digits = min_digits
        self.max_digits = max_digits
        self.canvas_size = canvas_size
        self.digit_size = digit_size

        print(f"Pre-generating {self.length} composite images... this may take a while.")
        self.canvases = []
        self.sequences = []
        for _ in tqdm(range(self.length)):
            canvas, sequence = get_composite_image_and_sequence(self.dataset, self.min_digits, self.max_digits, self.canvas_size, self.digit_size)
            self.canvases.append(torch.tensor(canvas, dtype=torch.float).unsqueeze(0))
            self.sequences.append(torch.tensor(sequence, dtype=torch.long))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.canvases[idx], self.sequences[idx]

def sequence_collate_fn(batch, max_seq_length = 10, pad_token_id=-1):
    """
    Collate function for autoregressive sequence training.
    
    Transforms (image, sequence) pairs into:
    - INPUT: IMAGE, [START_TOKEN, s1, s2, s3, s4]  
    - OUTPUT: [s1, s2, s3, s4, STOP_TOKEN]
    The sequence is padded to the max_seq_length (including the START_TOKEN and STOP_TOKEN).
    """
    images = []
    input_sequences = []
    output_sequences = []

    START_TOKEN = 10
    STOP_TOKEN = 10

    for image, sequence in batch:
        images.append(image)
        input_sequence = torch.cat((torch.tensor([START_TOKEN], dtype=torch.long), sequence))
        output_sequence = torch.cat((sequence, torch.tensor([STOP_TOKEN], dtype=torch.long)))

        # Truncate or pad the sequences to the max_seq_length
        if len(input_sequence) > max_seq_length:
            input_sequence = input_sequence[..., :max_seq_length]
            output_sequence = output_sequence[..., :max_seq_length]
        else:
            input_sequence = torch.cat((input_sequence, torch.full((max_seq_length - len(input_sequence),), pad_token_id)))
            output_sequence = torch.cat((output_sequence, torch.full((max_seq_length - len(output_sequence),), pad_token_id)))

        input_sequences.append(input_sequence)
        output_sequences.append(output_sequence)

    images = torch.stack(images)
    input_sequences = torch.stack(input_sequences)
    output_sequences = torch.stack(output_sequences)

    return images, input_sequences, output_sequences