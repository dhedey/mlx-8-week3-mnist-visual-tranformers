from .create_composite import get_composite_image_and_sequence, load_mnist_dataset
from torch.utils.data import Dataset
import torch
import torchvision
import numpy as np
import os
import time
import einops
from tqdm import tqdm
import hashlib
import pickle
import wandb
from .wandb_config import WANDB_ENTITY, WANDB_PROJECT_NAME

class BesCombine(Dataset):
  def __init__(self, max_sequence_length: int = 17, pad_token_id: int = -1, start_token_id: int = 10, stop_token_id: int = 10, train=True, h_patches = 2, w_patches = 2, length = None, p_skip = 0):
    super().__init__()
    #guarantee that h_patches and w_patches are integ
    self.h_patches = h_patches
    self.w_patches = w_patches
    self.p_skip = p_skip

    self.pad_token_id = pad_token_id
    self.start_token_id = start_token_id
    self.stop_token_id = stop_token_id

    if max_sequence_length < h_patches * w_patches + 1:
       raise ValueError("The model's max_sequence_length must be at least h_patches * w_patches + 1")
    self.max_sequence_length = max_sequence_length

    self.tk = { '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 's': 10}
    gen = np.random.default_rng(42) if not train else np.random.default_rng(int(time.time()))
    data_folder = os.path.join(os.path.dirname(__file__), "datasets")
    ds = torchvision.datasets.MNIST(root=data_folder, train=train, download=True)
    self.ln = len(ds) if length is None else int(length)

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
  
  def collate_fn(self, batch):
    return sequence_collate_fn(
        batch,
        max_seq_length=self.max_sequence_length,
        pad_token_id=self.pad_token_id,
        start_token_id=self.start_token_id,
        stop_token_id=self.stop_token_id
    )


class CompositeDataset(Dataset):
    def __init__(self, train: bool = True, pad_token_id: int = -1, start_token_id: int = 10, stop_token_id: int = 10, dataset = None, length = 100000, min_digits = 1, max_digits = 5, canvas_size=(256, 256), digit_size=28):
        if dataset is None:
            self.dataset = load_mnist_dataset(train=train)
        else:
            self.dataset = dataset
        self.length = length
        self.min_digits = min_digits
        self.max_digits = max_digits
        self.canvas_size = canvas_size
        self.digit_size = digit_size

        self.pad_token_id = pad_token_id
        self.start_token_id = start_token_id
        self.stop_token_id = stop_token_id

        # Generate cache key based on parameters
        self.cache_key = self._generate_cache_key()
        self.local_cache_path = self._get_local_cache_path()
        
        # Try to load from cache using three-tier strategy
        self.canvases, self.sequences = self._load_or_create_dataset()

    def _generate_cache_key(self) -> str:
        """Generate a unique cache key based on dataset parameters."""
        params = {
            'train': hasattr(self.dataset, 'train') and self.dataset.train,
            'length': self.length,
            'min_digits': self.min_digits,
            'max_digits': self.max_digits,
            'canvas_size': self.canvas_size,
            'digit_size': self.digit_size,
        }
        
        # Create hash from parameters
        param_str = str(sorted(params.items()))
        return hashlib.md5(param_str.encode()).hexdigest()[:12]
    
    def _get_local_cache_path(self) -> str:
        """Get the local cache file path."""
        cache_dir = os.path.join(os.path.dirname(__file__), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"composite_dataset_{self.cache_key}.pkl")
    
    def _get_wandb_artifact_name(self) -> str:
        """Get the wandb artifact name for this dataset."""
        train_suffix = "train" if (hasattr(self.dataset, 'train') and self.dataset.train) else "test"
        return f"composite-dataset-{train_suffix}-{self.cache_key}"
    
    def _load_from_local_cache(self) -> tuple[list, list]:
        """Try to load dataset from local cache."""
        if os.path.exists(self.local_cache_path):
            print(f"📁 Loading dataset from local cache: {self.local_cache_path}")
            try:
                with open(self.local_cache_path, 'rb') as f:
                    data = pickle.load(f)
                    if len(data['canvases']) == self.length and len(data['sequences']) == self.length:
                        print(f"✅ Successfully loaded {len(data['canvases'])} samples from local cache")
                        return data['canvases'], data['sequences']
                    else:
                        print(f"⚠️ Local cache has {len(data['canvases'])} samples, but need {self.length}. Will regenerate.")
            except Exception as e:
                print(f"⚠️ Failed to load from local cache: {e}")
        return None, None
    
    def _load_from_wandb(self) -> tuple[list, list]:
        """Try to load dataset from wandb artifact."""
        artifact_name = self._get_wandb_artifact_name()
        full_artifact_name = f"{WANDB_ENTITY}/{WANDB_PROJECT_NAME}/{artifact_name}:latest"
        
        print(f"🌐 Trying to download dataset from wandb: {full_artifact_name}")
        
        try:
            # Initialize wandb API (doesn't start a run)
            api = wandb.Api()
            artifact = api.artifact(full_artifact_name)
            
            print(f"📥 Downloading artifact from wandb...")
            download_dir = artifact.download()
            
            # Look for the pickle file in the downloaded directory
            pickle_files = [f for f in os.listdir(download_dir) if f.endswith('.pkl')]
            if not pickle_files:
                print(f"⚠️ No pickle file found in wandb artifact")
                return None, None
            
            pickle_path = os.path.join(download_dir, pickle_files[0])
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
                
            if len(data['canvases']) == self.length and len(data['sequences']) == self.length:
                print(f"✅ Successfully loaded {len(data['canvases'])} samples from wandb")
                
                # Save to local cache for future use
                print(f"💾 Saving to local cache for future use...")
                self._save_to_local_cache(data['canvases'], data['sequences'])
                
                return data['canvases'], data['sequences']
            else:
                print(f"⚠️ Wandb artifact has {len(data['canvases'])} samples, but need {self.length}. Will regenerate.")
                
        except Exception as e:
            print(f"⚠️ Failed to load from wandb: {e}")
        
        return None, None
    
    def _save_to_local_cache(self, canvases: list, sequences: list) -> None:
        """Save dataset to local cache."""
        try:
            data = {'canvases': canvases, 'sequences': sequences}
            with open(self.local_cache_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"💾 Saved dataset to local cache: {self.local_cache_path}")
        except Exception as e:
            print(f"⚠️ Failed to save to local cache: {e}")
    
    def _generate_dataset(self) -> tuple[list, list]:
        """Generate a new dataset from scratch."""
        print(f"🔄 Generating new composite dataset with {self.length} samples... this may take a while.")
        
        canvases = []
        sequences = []
        
        for _ in tqdm(range(self.length), desc="Generating composite images"):
            canvas, sequence = get_composite_image_and_sequence(
                self.dataset, 
                self.min_digits, 
                self.max_digits, 
                self.canvas_size, 
                self.digit_size
            )
            canvases.append(torch.tensor(canvas, dtype=torch.float).unsqueeze(0))
            sequences.append(torch.tensor(sequence, dtype=torch.long))
        
        return canvases, sequences
    
    def _load_or_create_dataset(self) -> tuple[list, list]:
        """Three-tier loading strategy: local cache -> wandb -> generate new."""
        
        # Tier 1: Try local cache
        canvases, sequences = self._load_from_local_cache()
        if canvases is not None:
            return canvases, sequences
        
        # Tier 2: Try wandb
        canvases, sequences = self._load_from_wandb()
        if canvases is not None:
            return canvases, sequences
        
        # Tier 3: Generate new dataset
        print(f"🆕 No cached version found. Generating new dataset...")
        canvases, sequences = self._generate_dataset()
        
        # Save to local cache
        self._save_to_local_cache(canvases, sequences)
        
        return canvases, sequences

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.canvases[idx], self.sequences[idx]
    
    def collate_fn(self, batch):
        return sequence_collate_fn(
            batch,
            max_seq_length=self.max_digits + 1,
            pad_token_id=self.pad_token_id,
            start_token_id=self.start_token_id,
            stop_token_id=self.stop_token_id
        )

def sequence_collate_fn(batch, max_seq_length = 10, pad_token_id=-1, start_token_id=10, stop_token_id=10):
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

    for image, sequence in batch:
        images.append(image)
        input_sequence = torch.cat((torch.tensor([start_token_id], dtype=torch.long), sequence))
        output_sequence = torch.cat((sequence, torch.tensor([stop_token_id], dtype=torch.long)))

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