import numpy as np
import matplotlib.pyplot as plt

# True parameters of the linear model
true_b = 1
true_w = 2

# Random initialization of parameters
np.random.seed(42)
b = np.random.randn(1)  # Initial bias
w = np.random.randn(1)  # Initial weight
N = 100                 # Number of samples
lr = 0.1                # Learning rate
n_epochs = 1000         # Number of training epochs

# Generate synthetic data: y = 2*x + 1 + noise
X = np.random.rand(N, 1)
y = true_w * X + true_b + (0.1 * np.random.randn(N, 1))

# Shuffle and split into train (80%) and validation (20%)
idx = np.arange(N)
np.random.shuffle(idx)

X_train, y_train = X[:int(N * 0.8)], y[:int(N * 0.8)]
X_val, y_val = X[int(N * 0.8):], y[int(N * 0.8):]

# Training loop - Gradient Descent from scratch
losses = []
for epoch in range(n_epochs):
    # Forward pass: compute predictions
    yhat = X_train * w + b
    
    # Compute error and loss (MSE)
    error = yhat - y_train
    loss = (error ** 2).mean()
    
    # Compute gradients manually
    b_grad = 2 * error.mean()
    w_grad = 2 * (error * X_train).mean()
    
    # Update parameters using gradient descent
    b -= b_grad * lr
    w -= w_grad * lr
    
    # Store loss for plotting
    losses.append(loss)

# Print final learned parameters
print(f"Learned bias (b): {b[0]:.4f}, Learned weight (w): {w[0]:.4f}")
print(f"True values - bias: {true_b}, weight: {true_w}")

# Plot the loss curve
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training Loss over Epochs')
plt.grid(True, alpha=0.3)
plt.show()