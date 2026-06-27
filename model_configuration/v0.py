# Check if GPU is available, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 0.1

# Set seed for reproducibility
torch.manual_seed(42)
# Create model and immediately send it to the selected device
model = nn.Sequential(nn.Linear(1, 1)).to(device)

# Define SGD optimizer to update parameters
# Retrieve parameters directly from the model
optimizer = optim.SGD(model.parameters(), lr=lr)

# Define MSE loss function (mean squared error)
loss_fn = nn.MSELoss(reduction='mean')
