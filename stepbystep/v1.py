import random
import datetime

import numpy as np
import torch
import torch.backends
import torch.backends.mps
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms import Normalize
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

# A completely empty class
class StepByStep(object):
    def __init__(self, model, loss_fn, optimizer):
        # Define class attributes here

        # First, store the parameters as attributes for later use
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = 'mps'  # Using Apple Metal Performance Shaders (for M1/M2 Macs)
        self.model.to(self.device)
        self.train_loader = None
        self.val_loader = None
        self.writer = None

        # These attributes will be computed internally
        self.losses = []
        self.val_losses = []
        self.total_epochs = 0

        # hooks
        self.visualization = {}
        self.handlers = {}

        # Create train_step function for model, loss function, and optimizer
        # Note: no parameters, it uses class attributes directly
        self.train_step = self._make_train_step()
        # Create val_step function for model and loss function
        self.val_step = self._make_val_step()

    def to(self, device):
        # This method allows the user to specify a different device
        # It sets the corresponding attribute (used later in mini-batches) and sends the model to the device
        self.device = device
        self.model.to(self.device)

    def set_loaders(self, train_loader, val_loader=None):
        # This method allows the user to define which train_loader (and val_loader, optional) to use
        self.train_loader = train_loader
        self.val_loader = val_loader

    # def set_tensorboard(self, name, folder='runs'):
    #     # This method allows the user to create a SummaryWriter to interact with TensorBoard
    #     suffix = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    #     self.writer = SummaryWriter(f'{folder}/{name}_{suffix}')

    def _make_train_step(self):
        # This method does not need ARGS
        # It can refer to attributes: self.model, self.loss_fn, and self.optimizer

        # Build a function that performs one step in the training loop
        def perform_train_step(x, y):
            # Set model to training mode
            self.model.train()

            # Step 1: Compute model predictions - Forward pass
            yhat = self.model(x)
            # Step 2: Compute the loss
            loss = self.loss_fn(yhat, y)
            # Step 3: Compute gradients for parameters
            loss.backward()

            # Step 4: Update parameters using gradients and learning rate
            self.optimizer.step()
            self.optimizer.zero_grad()

            return loss.item()

        return perform_train_step

    def _make_val_step(self):
        # Build a function that performs one step in the validation loop
        def perform_val_step(x, y):
            # Set model to evaluation mode
            self.model.eval()

            # Step 1: Compute model predictions - Forward pass
            yhat = self.model(x)
            # Step 2: Compute the loss
            loss = self.loss_fn(yhat, y)

            return loss.item()

        return perform_val_step

    def _mini_batch(self, validation=False):
        # Mini-batch can be used with both loaders
        # The parameter 'validation' defines which loader will be used
        # and correspondingly which step function will be used
        if validation:
            data_loader = self.val_loader
            step = self.val_step
        else:
            data_loader = self.train_loader
            step = self.train_step

        if data_loader is None:
            return None

        # Set up data loader and step function
        # This is our previous mini-batch loop
        mini_batch_losses = []
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            mini_batch_loss = step(x_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)

        loss = np.mean(mini_batch_losses)

        return loss

    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        try:
            self.train_loader.sampler.generator.manual_seed(seed)
        except AttributeError:
            pass

    def train(self, n_epochs, seed=42):
        # Ensure reproducibility of the training process
        self.set_seed(seed)

        for epoch in range(n_epochs):
            # Track the number of epochs by updating the corresponding attribute
            self.total_epochs += 1

            # Inner loop
            # Perform training using mini-batches
            loss = self._mini_batch(validation=False)
            self.losses.append(loss)

            # Evaluation - no gradients needed during evaluation
            with torch.no_grad():
                # Perform evaluation using mini-batches
                val_loss = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)

            # If SummaryWriter is set...
            if self.writer is not None:
                scalars = {'training': loss}
                if self.val_losses is not None:
                    scalars.update({'validation': val_loss})

                self.writer.add_scalars(
                    main_tag="loss",
                    tag_scalar_dict=scalars,
                    global_step=epoch
                )
        if self.writer is not None:
            self.writer.flush()

    def save_checkpoint(self, filename):
        # Build a dictionary containing all elements to resume training
        checkpoint = {
            'epoch': self.total_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.losses,
            'val_loss': self.val_losses
        }

        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        # Load the dictionary
        checkpoint = torch.load(filename)

        # Restore model and optimizer states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.total_epochs = checkpoint['epoch']
        self.losses = checkpoint['loss']
        self.val_losses = checkpoint['val_loss']

        self.model.train()  # always use TRAIN for resuming training

    def predict(self, x):
        # Set to evaluation mode for prediction
        self.model.eval()
        # Get NumPy input and make it a float tensor
        x_tensor = torch.as_tensor(x).float().to(self.device)
        # Send input to device and use model for prediction
        y_hat_tensor = self.model(x_tensor)
        # Set it back to training mode
        self.model.train()
        # Detach, bring it to CPU, and return to NumPy
        return y_hat_tensor.detach().cpu().numpy()

    def plot_losses(self):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.losses, label='Training Losses', c='b')
        if self.val_loader:
            plt.plot(self.val_losses, label='Validation Loss', c='r')
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        return fig

    def add_graph(self):
        if self.train_loader and self.writer:
            # Get a single mini-batch so add_graph can be used
            x_dummy, y_dummy = next(iter(self.train_loader))
            self.writer.add_graph(self.model, x_dummy.to(self.device))

    def count_parameters(self):
        return sum([p.numel() for p in self.model.parameters() if p.requires_grad])

    @staticmethod
    def _visualize_tensors(axes, x, y=None, yhat=None, layer_name='', title=None, img_value=True):
        # Number of images is the number of 'subplots' per row
        n_images = len(axes)
        # Get the max and min for scaling grayscale
        minv, maxv = np.min(x), np.max(x)
        # For each image
        for j, image in enumerate(x[:n_images]):
            ax = axes[j]
            # Set title, labels, and remove ticks
            if title is not None:
                ax.set_title(f'{title} #{j}', fontsize=12)
            shp = np.atleast_2d(image).shape
            ax.set_ylabel(
                f'{layer_name}\n{shp[0]}x{shp[1]}', rotation=0, labelpad=40)

            xlabel1 = '' if y is None else f"\nLabel: {y[j]}"
            xlabel2 = '' if yhat is None else f"\nPrediction: {yhat[j]}"
            xlabel = f'{xlabel1}{xlabel2}'
            if len(xlabel):
                ax.set_xlabel(xlabel, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

            # Plot weights as images
            ax.imshow(np.atleast_2d(image.squeeze()),
                      cmap='gray', vmin=minv, vmax=maxv)

            if img_value:
                # Display values in each square
                image_2d = np.atleast_2d(image.squeeze())
                for (ii, jj), val in np.ndenumerate(image_2d):
                    # ax.text coordinates are (x, y) where x is column index, y is row index
                    ax.text(jj, ii, f'{val:.2f}', ha='center',
                            va='center', color='red', fontsize=8)

    def visualize_filter(self, layer_name, **kwargs):
        """layer_name: name of the layer, kwargs: {"img_value": bool=True}"""
        try:
            # Get the layer object from the model
            layer = self.model
            for name in layer_name.split('.'):
                layer = getattr(layer, name)
            # Only focus on filters of 2D convolutions
            if isinstance(layer, nn.Conv2d):
                # Get weights
                weights = layer.weight.data.cpu().numpy()
                # weights -> (output channels (filters), input channels, H, W)
                n_filters, n_channels, _, _ = weights.shape

                # Build the image
                size = (2 * n_channels + 2, 2 * n_filters)
                fig, axes = plt.subplots(n_filters, n_channels, figsize=size)
                axes = np.atleast_2d(axes)

                # Iterate through each channel (filter)
                for i in range(n_filters):
                    StepByStep._visualize_tensors(
                        axes[i, :],
                        weights[i],
                        layer_name=f'Filter #{i}',
                        title='Channel',
                        img_value=kwargs['img_value']
                    )

                fig.tight_layout()
                for ax in axes.flat:
                    ax.label_outer()

                return fig
        except AttributeError:
            return

    def attach_hooks(self, layers_to_hook, hook_fn=None):
        # Clear any previous values
        self.visualization = {}
        # Create dictionary to map layer objects to their names
        modules = list(self.model.named_modules())
        layer_names = {layer: name for name, layer in modules}

        if hook_fn is None:
            def hook_fn(layer, inputs, outputs):
                # Get layer name
                name = layer_names[layer]
                # Detach output
                values = outputs.detach().cpu().numpy()
                # Since hook functions may be called multiple times
                # e.g., if predicting on multiple mini-batches
                # Handle concatenation of results
                if self.visualization[name] is None:
                    self.visualization[name] = values
                else:
                    self.visualization[name] = np.concatenate([
                        self.visualization[name], values
                    ])

        for name, layer in modules:
            # If layer is in the list
            if name in layers_to_hook:
                # Initialize the corresponding key in the dictionary
                self.visualization[name] = None
                # Register forward hook and keep handle in another dictionary
                self.handlers[name] = layer.register_forward_hook(hook_fn)

    def remove_hooks(self):
        # Remove all hooks
        for name, handler in self.handlers.items():
            handler.remove()
        # Clear the dictionary
        self.handlers = {}

    def visualize_outputs(self, layers, n_images=10, y=None, y_hat=None):
        layers = [l for l in layers if l in self.visualization.keys()]
        shapes = [self.visualization[layer].shape for layer in layers]

        n_rows = []
        for shape in shapes:
            if len(shape) == 4:  # 4D output (batch_size, channels, height, width)
                n_rows.append(shape[1])  # number of channels
            elif len(shape) == 2:  # 2D output (batch_size, features)
                n_rows.append(1)
            else:
                raise ValueError(f"Unsupported shape: {shape}")

        total_rows = np.sum(n_rows)

        fig, axes = plt.subplots(total_rows, n_images, figsize=(
            1.5 * n_images, 1.5 * total_rows))
        # axes = np.atleast_2d(axes)

        # Loop through layers, each layer gets one row of subplots
        current_row = 0
        for i, layer in enumerate(layers):
            # Get the generated feature maps for this layer
            output = self.visualization[layer]
            is_vector = len(output.shape) == 2

            for j in range(n_rows[i]):
                StepByStep._visualize_tensors(
                    axes[current_row, :],
                    output if is_vector else output[:, j].squeeze(),
                    y,
                    y_hat,
                    layer_name=layers[i],
                    title='Image' if current_row == 0 else None,
                    img_value=False
                )
                current_row += 1

        for ax in axes.flat:
            ax.label_outer()
        fig.tight_layout()

        return fig

    def correct(self, x, y, threshold=0.5):
        # Move input data and labels to the specified device
        x, y = x.to(self.device), y.to(self.device)

        # Make predictions in inference mode
        with torch.inference_mode():
            yhat = self.model(x)

        # Get batch size and number of classes
        # (only 1 if it's binary)
        n_samples, n_dims = yhat.shape

        if n_dims > 1:
            # In multi-class, the largest logit always wins
            # So no need to bother getting probabilities

            # This is PyTorch's argmax version
            # But it returns a tuple: (max value, index of max value)
            _, predicted = torch.max(yhat, 1)
        else:
            n_dims = 2  # binary classification

            # In binary classification, need to check if the last layer is sigmoid (then it produces probabilities)
            # if isinstance(self.model, nn.Sequential) and isinstance(self.model[-1], nn.Sigmoid):
            if yhat.min() >= 0 and yhat.max() <= 1:  # yhat is already probabilities
                predicted = (yhat > threshold).long()
            else:
                predicted = (F.sigmoid(yhat) > threshold).long()

        # How many samples were correctly classified per class
        result = []
        for c in range(n_dims):
            n_class = (y == c).sum().item()
            n_correct = (predicted[y == c] == c).sum().item()
            result.append((n_correct, n_class))

        return torch.tensor(result)

    @staticmethod
    def loader_apply(loader, func, reduce='sum'):
        results = [func(x, y) for i, (x, y) in enumerate(loader)]
        results = torch.stack(results, axis=0)

        if reduce == 'sum':
            results = results.sum(axis=0)
        elif reduce == 'mean':
            results = results.float().mean(axis=0)

        return results

    @staticmethod
    def statistics_per_channel(images, labels):
        # Get input image shape (n_samples, n_channels, n_height, n_width)
        n_samples, n_channels, n_height, n_width = images.size()

        # Flatten pixels per channel to 1D
        flatten_per_channel = images.reshape(n_samples, n_channels, -1)

        # Compute statistics per channel per image
        # Mean of pixels per channel (n_samples, n_channels)
        means = flatten_per_channel.mean(axis=2)
        # Standard deviation of pixels per channel (n_samples, n_channels)
        stds = flatten_per_channel.std(axis=2)

        # Compute statistics across all images in the mini-batch
        sum_means = means.sum(axis=0)  # Sum of pixel means per channel across all images (n_channels,)
        sum_stds = stds.sum(axis=0)    # Sum of pixel stds per channel across all images (n_channels,)
        n_samples = torch.tensor(
            [n_samples] * n_channels).float()  # Number of samples (n_channels,)

        # Stack statistics together (3, n_channels)
        return torch.stack([n_samples, sum_means, sum_stds], axis=0)

    @staticmethod
    def make_normalizer(loader):
        total_samples, total_means, total_stds = \
            StepByStep.loader_apply(loader, StepByStep.statistics_per_channel)
        norm_mean = total_means / total_samples
        norm_std = total_stds / total_samples

        return Normalize(mean=norm_mean, std=norm_std)