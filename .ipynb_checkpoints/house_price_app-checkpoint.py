import streamlit as st
import numpy as np
from linear_regression import LinearRegression
import matplotlib.pyplot as plt

st.title('House Price Predictor')
st.write('Predict house prices based on square footage using Linear Regression')

# Generate sample data (same as in house_price_prediction.py for consistency)
np.random.seed(42)
house_sizes = np.random.uniform(1000, 5000, 100)
house_prices = 200 + 0.3 * house_sizes + np.random.normal(0, 50, 100)

# Train the model
model = LinearRegression(learning_rate=0.0000001)
model.fit(house_sizes, house_prices)

# Create input for house size
house_size = st.slider('Select House Size (square feet)', 
                      min_value=1000, 
                      max_value=5000, 
                      value=2500,
                      step=100)

# Make prediction
predicted_price = model.predict(house_size)

# Display prediction
st.write(f'### Predicted House Price: ${predicted_price:,.2f}K')

# Create and display plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(house_sizes, house_prices, color='blue', alpha=0.5, label='Training Data')
X_test = np.array([1000, 5000])
y_pred = model.predict(X_test)
ax.plot(X_test, y_pred, color='red', label='Regression Line')

# Plot the current prediction point
ax.scatter([house_size], [predicted_price], color='green', s=100, label='Current Prediction')

ax.set_xlabel('House Size (sq ft) - Input Feature')
ax.set_ylabel('House Price ($K) - Target Variable')
ax.set_title('House Price vs Size Linear Regression')
ax.legend()

st.pyplot(fig)

# Display model information
st.write('### Model Information')
st.write(f'- Weight (Price per sq ft): ${model.weights:.2f}K')
st.write(f'- Bias (Base price): ${model.bias:.2f}K')

# Add explanation about the algorithm
st.write('### How it works')
st.write('''
This app uses a custom Linear Regression model trained on synthetic housing data. 
The model learns the relationship between house size (input feature) and price (target variable) 
using gradient descent optimization.
''')

st.write('### Mathematical Formula')
st.latex(r'''
\text{Price} = w \times \text{size} + b
''')
st.write('''
where:
- Price is the output/target variable
- w is the weight (price per square foot)
- size is the input feature
- b is the bias (base price)
''')

st.write('### Gradient Descent vs Backpropagation')
st.write('''
This linear regression model uses simple gradient descent, NOT backpropagation. Here's why:

1. Backpropagation is specifically used in neural networks with multiple layers to calculate gradients 
   through the chain rule, propagating errors backwards through the network's layers.

2. Linear regression is a single-layer model with a direct computation:
   - Forward pass: y = wx + b
   - Error calculation: MSE = (1/n) * Σ(y_true - y_pred)²
   - Direct gradient computation:
     * ∂(MSE)/∂w = (-2/n) * Σ(x * (y_true - y_pred))
     * ∂(MSE)/∂b = (-2/n) * Σ(y_true - y_pred)

3. Since there are no hidden layers and the computation is direct, we don't need the chain rule 
   or backpropagation. We can compute the gradients directly from the error with respect to our parameters.
''')

st.write('### Loss Function (Mean Squared Error)')
st.latex(r'''
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_{true} - y_{pred})^2
''')
st.write('''
The Mean Squared Error (MSE) measures how well our model fits the data by calculating
the average squared difference between predicted and actual house prices. The model
uses gradient descent to minimize this error by adjusting the weight (w) and bias (b)
parameters.
''')

# Plot loss history
st.write('### Training Loss Over Time')
fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
ax_loss.plot(model.loss_history)
ax_loss.set_xlabel('Epoch')
ax_loss.set_ylabel('Mean Squared Error')
ax_loss.set_title('Training Loss Over Time')
st.pyplot(fig_loss)

st.write('''
The graph above shows how the Mean Squared Error decreases during training,
indicating that the model is learning to better predict house prices by adjusting
its parameters (weight and bias) through gradient descent optimization.
''')
