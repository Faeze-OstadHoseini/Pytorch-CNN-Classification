
n_epochs = 200

losses = []

for epoch in range(n_epochs):
    # Perform one epoch of mini-batch training
    loss = mini_batch(device, train_loader, train_step)
    losses.append(loss)
