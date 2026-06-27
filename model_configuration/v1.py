# Check if GPU is available, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set learning rate
lr = 0.1

# Set seed for reproducibility
torch.manual_seed(42)

# Create model and immediately send it to the device
model = nn.Sequential(nn.Linear(1,1)).to(device)

# Define SGD optimizer to update parameters
optimizer = optim.SGD(model.parameters(), lr=lr)

# Define MSE loss function (mean squared error)
loss_fn = nn.MSELoss(reduction='mean')

# Create train_step function for model, loss function, and optimizer
train_step = make_train_step(model, loss_fn, optimizer)
