n_epochs = 200

# Lists to store training and validation losses
losses = []
val_losses = []

for epoch in range(n_epochs):
    # Training - inner loop over mini-batches
    loss = mini_batch(device, train_loader, train_step_fn)
    losses.append(loss)

    # Validation - evaluate on validation data
    with torch.no_grad():
        val_loss = mini_batch(device, val_loader, val_step_fn)
        val_losses.append(val_loss)

    # Record losses for each epoch under the main tag 'loss' in TensorBoard
    writer.add_scalars(
        main_tag='loss',
        tag_scalar_dict={'training':loss, 'validation':val_loss},
        global_step = epoch
    )

# Close the TensorBoard writer
writer.close()
