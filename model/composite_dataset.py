from .create_composite import get_composite_image_and_sequence, load_mnist_dataset
from torch.utils.data import Dataset
import torch

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

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        canvas, sequence = get_composite_image_and_sequence(self.dataset, 1, self.min_digits, self.max_digits, self.canvas_size, self.digit_size)
        return canvas[0], sequence[0]

def sequence_collate_fn(batch, max_seq_length = 10):
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
    STOP_TOKEN = 11
    PAD_TOKEN = -1

    for image, sequence in batch:
        images.append(image)
        input_sequence = torch.cat((torch.tensor([START_TOKEN], dtype=torch.long), sequence))
        output_sequence = torch.cat((sequence, torch.tensor([STOP_TOKEN], dtype=torch.long)))

        #truncate or pad the sequences to the max_seq_length
        if len(input_sequence) > max_seq_length:
            input_sequence = input_sequence[..., :max_seq_length]
            output_sequence = output_sequence[..., :max_seq_length]
        else:
            input_sequence = torch.cat((input_sequence, torch.full((max_seq_length - len(input_sequence),), PAD_TOKEN)))
            output_sequence = torch.cat((output_sequence, torch.full((max_seq_length - len(output_sequence),), PAD_TOKEN)))

        input_sequences.append(input_sequence)
        output_sequences.append(output_sequence)

    images = torch.stack(images)
    input_sequences = torch.stack(input_sequences)
    output_sequences = torch.stack(output_sequences)

    return images, input_sequences, output_sequences


        