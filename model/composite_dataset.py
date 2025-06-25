from .create_composite import get_composite_image_and_sequence, load_mnist_dataset
from torch.utils.data import Dataset
import torch
import torchvision
import random
import einops

class BesCombine(Dataset):
  def __init__(self, train=True):
    super().__init__()
    self.tf = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    self.tk = { '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 's': 10}
    self.ds = torchvision.datasets.MNIST(root='.', train=train, download=True)
    self.ti = torchvision.transforms.ToPILImage()
    self.ln = len(self.ds)

  def __len__(self):
    return len(self.ds)

  def __getitem__(self, idx):
    idx = random.sample(range(self.ln), 4)
    store = [self.ds[i][0] for i in idx]
    label = [self.ds[i][1] for i in idx]
    tnsrs = [self.tf(img) for img in store]
    stack = torch.stack(tnsrs, dim=0).squeeze()
    combo = einops.rearrange(stack, '(h w) ph pw -> (h ph) (w pw)', h=2, w=2, ph=28, pw=28)
    #patch = einops.rearrange(combo, '(h ph) (w pw) -> (h w) ph pw', ph=14, pw=14)
    #label = [10] + label + [11]
    return combo, torch.tensor(label)


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
        canvas, sequence = get_composite_image_and_sequence(self.dataset, self.min_digits, self.max_digits, self.canvas_size, self.digit_size)
        return torch.tensor(canvas, dtype=torch.float), torch.tensor(sequence, dtype=torch.long)

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


        