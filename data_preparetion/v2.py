# Set seed for reproducibility in PyTorch
torch.manual_seed(13)

# Build tensors from numpy arrays before splitting
x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()

# Build dataset with all data points
dataset = TensorDataset(x_tensor, y_tensor)

# Perform train/validation split
ratio = .8
n_total = len(dataset)
n_train = int(n_total * ratio)
n_val = n_total - n_train

train_data, val_data = random_split(dataset, [n_train, n_val])

# Build loaders for each set
train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)  # Shuffle for training
val_loader = DataLoader(dataset=val_data, batch_size=16) # No shuffle for validation
