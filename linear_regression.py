import numpy as np

"""
Linear regression is a parametric learning algorithm, which has fixed number of parameters
"""
class LinearRegression:
    """
    Linear Regression Implementation using Gradient Descent

    Mathematical Formula:
    y = wx + b
    where:
    - y is the predicted value (house price)
    - w is the weight/coefficient (price per square foot)
    - x is the input feature (house size)
    - b is the bias/intercept

    Loss Function (Mean Squared Error):
    MSE = (1/n) * Σ(y_true - y_pred)²
    where:
    - n is the number of samples
    - y_true is the actual value
    - y_pred is the predicted value (wx + b)

    Gradient Descent Update Rules:
    For each iteration:
    w = w - α * ∂(MSE)/∂w
    b = b - α * ∂(MSE)/∂b

    where:
    - α is the learning rate
    - ∂(MSE)/∂w is the partial derivative of MSE with respect to w
    - ∂(MSE)/∂b is the partial derivative of MSE with respect to b

    Partial Derivatives Derivation:
    1. Start with MSE = (1/n) * Σ(y_true - y_pred)²
    2. Substitute y_pred = wx + b:
       MSE = (1/n) * Σ(y_true - (wx + b))²
    
    3. For weight (w) derivative:
       ∂(MSE)/∂w = ∂/∂w[(1/n) * Σ(y_true - (wx + b))²]
       = (1/n) * Σ(2 * (y_true - (wx + b)) * (-x))
       = (-2/n) * Σ(x * (y_true - y_pred))
       The -2 comes from:
       - The chain rule giving us 2 * (y_true - (wx + b))
       - The derivative of (wx + b) with respect to w giving us x
       - Combining these terms: 2 * (y_true - (wx + b)) * (-x)
    
    4. For bias (b) derivative:
       ∂(MSE)/∂b = ∂/∂b[(1/n) * Σ(y_true - (wx + b))²]
       = (1/n) * Σ(2 * (y_true - (wx + b)) * (-1))
       = (-2/n) * Σ(y_true - y_pred)
       The -2 comes from:
       - The chain rule giving us 2 * (y_true - (wx + b))
       - The derivative of (wx + b) with respect to b giving us 1
       - Combining these terms: 2 * (y_true - (wx + b)) * (-1)
    """

    def __init__(self, learning_rate=0.01):
        """
        Initialize the model with a learning rate for gradient descent.
        A smaller learning rate means more stable but slower learning.
        A larger learning rate means faster but potentially unstable learning.
        """
        self.learning_rate = learning_rate
        self.weights = None  # Coefficient (price per square foot)
        self.bias = None  # Intercept (base price)

    def _calculate_gradients(self, X, y, y_pred, n_samples):
        """
        Calculate the partial derivatives (gradients) for weights and bias using batch gradient descent.
        
        The -2 in the gradient formulas comes from taking the derivative of the squared error loss:
        1. For weights: When we take ∂/∂w of (y_true - (wx + b))², we get:
           2 * (y_true - (wx + b)) * (-x) = -2x * (y_true - y_pred)
        2. For bias: When we take ∂/∂b of (y_true - (wx + b))², we get:
           2 * (y_true - (wx + b)) * (-1) = -2 * (y_true - y_pred)
        
        Args:
            X: Input features
            y: True target values
            y_pred: Predicted values
            n_samples: Number of samples in the dataset
            
        Returns:
            tuple: (weight_gradient, bias_gradient)
            - weight_gradient: ∂(MSE)/∂w = (-2/n) * Σ(x * (y_true - y_pred))
            - bias_gradient: ∂(MSE)/∂b = (-2/n) * Σ(y_true - y_pred)
        """
        weight_gradient = (-2 / n_samples) * np.sum(X * (y - y_pred))
        bias_gradient = (-2 / n_samples) * np.sum(y - y_pred)
        return weight_gradient, bias_gradient

    def _stochastic_gradient(self, X, y, y_pred, idx):
        """
        Calculate the partial derivatives (gradients) for weights and bias using stochastic gradient descent.
        Instead of using the entire dataset, it uses a single random sample for each update.
        
        The -2 in the gradient formulas has the same mathematical origin as in batch gradient descent,
        but applied to a single sample instead of the average over all samples:
        1. For weights: -2 * x * (y_true - y_pred) for the single sample
        2. For bias: -2 * (y_true - y_pred) for the single sample
        
        Args:
            X: Input features array
            y: True target values array
            y_pred: Predicted values array
            idx: Index of the current sample
            
        Returns:
            tuple: (weight_gradient, bias_gradient)
            - weight_gradient: -2 * x * (y_true - y_pred) for single sample
            - bias_gradient: -2 * (y_true - y_pred) for single sample
        """
        # Calculate gradients for a single sample
        weight_gradient = -2 * X[idx] * (y[idx] - y_pred[idx])
        bias_gradient = -2 * (y[idx] - y_pred[idx])
        return weight_gradient, bias_gradient

    def fit(self, X, y, epochs=1000, method='batch'):
        """
        Train the model using gradient descent optimization.

        Args:
            X: Input features (house sizes)
            y: Target values (house prices)
            epochs: Number of training iterations
            method: 'batch' for batch gradient descent or 'stochastic' for stochastic gradient descent

        The training process:
        1. Initialize parameters (w and b) to zero
        2. For each epoch:
        a. Calculate predictions using current w and b
        b. Calculate gradients (partial derivatives)
        c. Update w and b using gradient descent
        d. Track the loss for monitoring convergence
        """
        # Initialize parameters
        n_samples = len(X)
        self.weights = 0  # Initialize price per square foot
        self.bias = 0  # Initialize base price

        # Training history for visualization
        self.loss_history = []

        # Gradient descent
        for epoch in range(epochs):
            # Forward pass: Calculate predictions
            y_pred = self.weights * X + self.bias

            if method == 'batch':
                # Batch gradient descent - use entire dataset
                weight_gradient, bias_gradient = self._calculate_gradients(X, y, y_pred, n_samples)
                
                # Update parameters using gradient descent
                self.weights -= self.learning_rate * weight_gradient
                self.bias -= self.learning_rate * bias_gradient

            else:  # stochastic gradient descent
                # Randomly shuffle data for this epoch
                indices = np.random.permutation(n_samples)
                
                # Update parameters for each sample
                for idx in indices:
                    # Calculate gradients for single sample
                    weight_gradient, bias_gradient = self._stochastic_gradient(X, y, y_pred, idx)
                    
                    # Update parameters using gradient descent
                    self.weights -= self.learning_rate * weight_gradient
                    self.bias -= self.learning_rate * bias_gradient
                    
                    # Update predictions after each parameter update
                    y_pred = self.weights * X + self.bias

            # Compute loss (MSE) for monitoring
            loss = np.mean((y - y_pred) ** 2)
            self.loss_history.append(loss)

    def predict(self, X):
        """
        Make predictions using the trained model.

        The prediction formula is:
        y = wx + b
        where:
        - y is the predicted house price - output/target variable
        - w is the learned price per square foot (weight)
        - x is the house size - input/features
        - b is the base price (bias)
        """
        return self.weights * X + self.bias
