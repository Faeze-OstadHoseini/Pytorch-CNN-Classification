# Define number of training epochs
n_epochs = 1000

for epoch in range(n_epochs):
    # Set model to training mode
    model.train()

    # Step 1: Compute model predictions - Forward pass
    yhat = model(x_train_tensor)

    # Step 2: Compute the loss
    loss = loss_fn(yhat, y_train_tensor)

    # Step 3: Compute gradients for parameters (w and b)
    loss.backward()

    # Step 3: Compute gradients for parameters (w and b)
    optimizer.step()
    optimizer.zero_grad()
