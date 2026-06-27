
device = 'cude' if torch.cuda.is_available() else 'cpu'

lr = 0.1

torch.manual_seed(42)

model = nn.Sequential(nn.Linear(1, 1)).to(device)

optimizer = optim.SGD(model.parameters(), lr=lr)

loss_fn = nn.MSELoss(reduction='mean')

# Create train_step function for model, loss function, and optimizer
train_step = make_train_step(model, loss_fn, optimizer)

# Create val_step function for model and loss function (no optimizer!)
val_step = make_val_step(model, loss_fn)
