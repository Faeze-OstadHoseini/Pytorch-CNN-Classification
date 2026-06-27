device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device:{device}')

# Data is currently in NumPy arrays
# Need to convert them to PyTorch tensors
# Then send them to the selected device (GPU/CPU)
x_train_tensor = torch.as_tensor(x_train).float().to(device)
y_train_tensor = torch.as_tensor(y_train).float().to(device)
