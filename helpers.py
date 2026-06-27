import numpy as np
import torch
from torch.utils.data import random_split, Dataset, WeightedRandomSampler
import matplotlib.pyplot as plt


class TransformedTensorDataset(Dataset):
    """Custom Dataset that applies transforms to tensor data"""
    def __init__(self, x, y, transform=None):
        super().__init__()
        self.x = x          # Feature tensor
        self.y = y          # Label tensor
        self.transform = transform  # Optional transform to apply

    def __getitem__(self, index):
        x = self.x[index]
        if self.transform:
            x = self.transform(x)
        y = self.y[index]
        return x, y

    def __len__(self):
        return len(self.x)


def index_splitter(n, splits, seed=13):
    """
    Split indices into train/validation sets
    
    Args:
        n: Total number of samples
        splits: List of split sizes (e.g., [80, 20] for 80-20 split)
        seed: Random seed for reproducibility
    """
    idx = torch.arange(n)

    # Convert splits to tensor and calculate multiplier
    split_tensor = torch.as_tensor(splits)
    multiplier = n // split_tensor.sum()
    split_tensor = (multiplier * split_tensor).long()

    # Handle any remainder by adding to first split
    diff = n - split_tensor.sum()
    split_tensor[0] += diff

    torch.manual_seed(seed)
    return random_split(idx, split_tensor)


def make_balanced_sampler(y):
    """
    Create a WeightedRandomSampler to balance imbalanced datasets
    
    Args:
        y: Labels tensor
        
    Returns:
        WeightedRandomSampler with balanced class weights
    """
    y_tensor = torch.as_tensor(y)

    # Count samples per class
    classes, count = y_tensor.unique(return_counts=True)
    # Calculate weights (inverse of class frequencies)
    weights = 1. / count.float()

    # Map each sample to its class index
    y_weights_indices = torch.searchsorted(classes, y_tensor)

    # Assign weights to each sample
    sample_weights = weights[y_weights_indices]

    generator = torch.Generator()
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(y),
        replacement=True,
        generator=generator,
    )


def plot_images(images, labels, n_plots=30, n_col=6):
    """
    Plot a grid of images with their labels
    
    Args:
        images: Image tensor
        labels: Label tensor
        n_plots: Number of images to display
        n_col: Number of columns in the grid
    """
    # Prevent n_plots from exceeding available images
    n_plots = min(n_plots, len(images))

    # Calculate rows and figure size
    n_rows = n_plots // n_col + (n_plots % n_col > 0)
    cols = min(1.5 * n_col, 15)

    fig, axes = plt.subplots(
        n_rows, n_col, figsize=(cols, 1.5 * n_rows))
    axes = np.atleast_2d(axes)

    # Plot each image
    for i, (image, label) in enumerate(zip(images[:n_plots], labels[:n_plots])):
        ax = axes[i // n_col, i % n_col]
        ax.imshow(image.squeeze(), cmap='gray')
        ax.set_title(f"#{i} - {label}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.label_outer()

    plt.tight_layout()
    return fig


class TransformedTensorDataset(Dataset):
    """Custom Dataset that applies transforms to tensor data (redefined)"""
    def __init__(self, x, y, transform=None):
        super().__init__()
        self.x = x          # Feature tensor
        self.y = y          # Label tensor
        self.transform = transform  # Optional transform to apply

    def __getitem__(self, index):
        x = self.x[index]

        if self.transform is not None:
            x = self.transform(x)
        return x, self.y[index]

    def __len__(self):
        return len(self.x)