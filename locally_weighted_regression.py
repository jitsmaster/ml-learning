import numpy as np
import matplotlib.pyplot as plt


# LOWESS is a non-parametric learning algorithm. The size of it's data grow linearly with the size training set
# Should only use it for small set of training data for predictions. When dataset is too large, this will take a lot of CPU and memory
class LocallyWeightedRegression:
    def __init__(self, tau=0.1):
        """
        Initialize Locally Weighted Regression model.

        Parameters:
        tau (float): The bandwidth parameter that controls the size of the local region
                    Larger tau = smoother fit (more bias, less variance)
                    Smaller tau = more flexible fit (less bias, more variance)
        """
        self.tau = tau
        self.X = None
        self.y = None

    def gaussian_kernel(self, x0, X):
        """
        Compute Gaussian kernel weights for each point in X relative to x0.

        Parameters:
        x0: Point at which to evaluate the regression
        X: Training data points

        Returns:
        weights: Array of weights for each training point
        """
        # Step 1: Calculate the squared Euclidean distance
        squared_distances = np.sum((X - x0) ** 2, axis=1)

        # Step 2: Scale the distances by the bandwidth parameter
        # This decided the width of Gaussian curve.
        scaled_distances = squared_distances / (2 * self.tau**2)

        # Step 3: Apply the exponential function
        weights = np.exp(-scaled_distances)

        # Alternatively, you can keep it as a one-liner:
        # Mathematical formula:
        # w(x) = exp(-Σ(x_i - x_0)^2 / (2 * τ^2))
        # where:
        #   w(x): weight for point x
        #   x_i: each dimension of the input point
        #   x_0: each dimension of the point we're calculating for
        #   τ (tau): bandwidth parameter
        #   Σ: sum over all dimensions
        # return np.exp(-np.sum((X - x0) ** 2, axis=1) / (2 * self.tau ** 2))

        return weights

    def fit(self, X, y):
        """
        Store training data for later use in predictions.
        This is what non-parametric algorithms do, real time using training data
        Memory intensive

        Parameters:
        X: Training features
        y: Target values
        """
        self.X = np.array(X)
        self.y = np.array(y)

    def predict(self, X_test):
        """
        Make predictions for test points using locally weighted regression.

        Parameters:
        X_test: Test points to make predictions for

        Returns:
        predictions: Predicted values for X_test
        """
        X_test = np.array(X_test)
        predictions = []

        # Make prediction for each test point
        for x0 in X_test:
            # Compute weights for all training points
            weights = self.gaussian_kernel(x0, self.X)

            # Create diagonal weight matrix
            W = np.diag(weights)

            # Add bias term to training data
            X_b = np.c_[np.ones(len(self.X)), self.X]

            # Compute weighted parameters (normal equation with weights)
            try:
                # Compute weighted parameters using the normal equation:
                # θ = (X^T W X)^(-1) X^T W y
                # Where:
                #   θ : parameter vector
                #   X : design matrix (including bias term)
                #   W : diagonal weight matrix
                #   y : target values
                #   ^T : matrix transpose
                #   ^(-1) : matrix inverse
                # @ is a matrix multiplication operator

                # Matrix multiplication is an operation that takes two matrices and produces another matrix. 
                # The operation involves multiplying the rows of the first matrix by the columns of the second 
                # matrix and summing the products.
                theta = np.linalg.inv(X_b.T @ W @ X_b) @ X_b.T @ W @ self.y
            except np.linalg.LinAlgError:
                # If matrix is singular, use pseudoinverse (pinv)
                theta = np.linalg.pinv(X_b.T @ W @ X_b) @ X_b.T @ W @ self.y

            # Make prediction (don't forget bias term)
            x0_b = np.r_[1, x0]
            prediction = x0_b @ theta
            predictions.append(prediction)

        return np.array(predictions)


# # Example usage and visualization
# if __name__ == "__main__":
#     # Generate synthetic data
#     np.random.seed(42)
#     X = np.linspace(-3, 3, 100)
#     y = np.sin(X) + np.random.normal(0, 0.1, 100)  # Sine wave with noise

#     # Reshape X for sklearn compatibility
#     X = X.reshape(-1, 1)

#     # Create and fit the model
#     model = LocallyWeightedRegression(tau=0.1)
#     model.fit(X, y)

#     # Generate test points and predictions
#     X_test = np.linspace(-3, 3, 300).reshape(-1, 1)
#     y_pred = model.predict(X_test)

#     # Plot the results
#     plt.figure(figsize=(10, 6))
#     plt.scatter(X, y, color="blue", alpha=0.5, label="Training data")
#     plt.plot(X_test, y_pred, color="red", label="LOWESS regression")
#     plt.title("Locally Weighted Regression")
#     plt.xlabel("X")
#     plt.ylabel("y")
#     plt.legend()
#     plt.grid(True)
#     plt.show()
