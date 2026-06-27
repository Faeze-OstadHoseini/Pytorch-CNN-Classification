
# Define number of training epochs
n_epochs = 1000

# List to store average loss per epoch
losses = []

for epoch in range(n_epochs):
    # Inner loop: iterate over mini-batches
    mini_batch_losses = []
    for x_batch, y_batch in train_loader:
        # Data exists on CPU, mini-batches are also on CPU
        # Need to send these mini-batches to the device where the model is
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # Perform one training step
        # and return the loss for this mini-batch
        mini_batch_loss = train_step(x_batch, y_batch)
        mini_batch_losses.append(mini_batch_loss)

    # Compute average loss across all mini-batches - this is the epoch loss
    loss = np.mean(mini_batch_losses)

    losses.append(loss)
