
# Data is in NumPy arrays
# Need to convert to PyTorch tensors
x_train_tensor = torch.from_numpy(x_train).float()
y_train_tensor = torch.from_numpy(y_train).float()

# Create Dataset (pairs x and y together)
train_data = TensorDataset(x_train_tensor, y_train_tensor)

# Create DataLoader for batch training
# - batch_size: 16 samples per batch
# - shuffle: randomize order each epoch
train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
