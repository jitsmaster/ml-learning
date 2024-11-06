import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression

"""
Note on Gradient Descent vs Backpropagation:

This linear regression model uses simple gradient descent, NOT backpropagation.
Here's why:

1. Backpropagation is specifically used in neural networks with multiple layers
   to calculate gradients through the chain rule, propagating errors backwards
   through the network's layers.

2. Linear regression is a single-layer model with a direct computation:
   - Forward pass: y = wx + b
   - Error calculation: MSE = (1/n) * Σ(y_true - y_pred)²
   - Direct gradient computation: 
     * ∂(MSE)/∂w = (-2/n) * Σ(x * (y_true - y_pred))
     * ∂(MSE)/∂b = (-2/n) * Σ(y_true - y_pred)

3. Since there are no hidden layers and the computation is direct,
   we don't need the chain rule or backpropagation. We can compute
   the gradients directly from the error with respect to our parameters.
"""

# Generate sample data
np.random.seed(42)
house_sizes = np.random.uniform(1000, 5000, 100)  # Square feet
# True relationship: price = 200K + 0.3K per sq ft + noise
house_prices = 200 + 0.3 * house_sizes + np.random.normal(0, 50, 100)  # Price in thousands

# Create and train the model
model = LinearRegression(learning_rate=0.0000001)  # Small learning rate for stability
model.fit(house_sizes, house_prices)

# Plotting
plt.figure(figsize=(12, 8))

# Plot training data
plt.scatter(house_sizes, house_prices, color='blue', alpha=0.5, label='Training Data')

# Plot regression line
X_test = np.array([1000, 5000])
y_pred = model.predict(X_test)
plt.plot(X_test, y_pred, color='red', label='Regression Line')

plt.xlabel('House Size (sq ft)')
plt.ylabel('House Price ($K)')
plt.title('House Price vs Size Linear Regression')
plt.legend()

# Plot loss history
plt.figure(figsize=(10, 6))
plt.plot(model.loss_history)
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Training Loss Over Time')

# Make some predictions
test_sizes = np.array([1500, 2500, 3500])
predictions = model.predict(test_sizes)

print("\nPredictions:")
for size, price in zip(test_sizes, predictions):
    print(f"House size: {size:.0f} sq ft -> Predicted price: ${price:.2f}K")

plt.show()
