
# Define number of training epochs
n_epochs = 1000

# List to store loss values for each epoch
losses = []

# For each epoch...
for epoch in range(n_epochs):
    # Perform a training step and return the corresponding loss
    loss = train_step(x_train_tensor, y_train_tensor)
    losses.append(loss)
