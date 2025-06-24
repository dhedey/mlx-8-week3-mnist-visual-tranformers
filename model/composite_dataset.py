from .create_composite import generate_composite_batch, load_mnist_dataset
from torch.utils.data import Dataset

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
        canvas, sequence = generate_composite_batch(self.dataset, 1, self.min_digits, self.max_digits, self.canvas_size, self.digit_size)
        return canvas[0], sequence[0]

    def __getitems__(self, indices):
        canvas, sequence = generate_composite_batch(self.dataset, len(indices), self.min_digits, self.max_digits, self.canvas_size, self.digit_size)
        return canvas, sequence


