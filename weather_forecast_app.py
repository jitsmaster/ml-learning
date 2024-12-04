import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from locally_weighted_regression import LocallyWeightedRegression
from datetime import datetime, timedelta

# Set page config
st.set_page_config(page_title="Weather Forecast Demo", layout="wide")

# Title and description
st.title("Weather Forecast using Locally Weighted Regression")
st.write("""
This demo shows how Locally Weighted Regression (LWR) can be used for weather forecasting.
The model learns local patterns in temperature data to make predictions.
""")

# Generate synthetic weather data
def generate_weather_data(days=30):
    np.random.seed(42)
    # Generate time points (1 measurement every hour)
    hours = days * 24
    time_points = np.linspace(0, hours, hours)
    
    # Generate synthetic temperature data with daily cycles and some noise
    daily_cycle = 10 * np.sin(2 * np.pi * time_points / 24)  # Daily temperature cycle
    trend = 0.1 * time_points  # Slight upward trend
    noise = np.random.normal(0, 2, hours)  # Random variations
    
    temperatures = 20 + daily_cycle + trend/50 + noise  # Base temperature of 20°C
    
    # Create timestamps
    start_date = datetime.now() - timedelta(days=days)
    dates = [start_date + timedelta(hours=i) for i in range(hours)]
    
    return pd.DataFrame({
        'timestamp': dates,
        'hour': time_points,
        'temperature': temperatures
    })

# Sidebar controls
st.sidebar.header("Model Parameters")
tau = st.sidebar.slider("Bandwidth (τ)", 
                       min_value=1.0, 
                       max_value=50.0, 
                       value=10.0, 
                       help="Controls how much influence distant points have on the prediction")

days = st.sidebar.slider("Number of days", 
                        min_value=5, 
                        max_value=60, 
                        value=30)

# Generate data
df = generate_weather_data(days)

# Prepare data for modeling
X = df['hour'].values.reshape(-1, 1)
y = df['temperature'].values

# Create and fit the model
model = LocallyWeightedRegression(tau=tau)
model.fit(X, y)

# Generate predictions
X_test = np.linspace(0, max(X)[0], 200).reshape(-1, 1)
y_pred = model.predict(X_test)

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(X, y, color='blue', alpha=0.5, label='Actual Temperatures')
ax.plot(X_test, y_pred, color='red', label='LWR Prediction')
ax.set_xlabel('Hours')
ax.set_ylabel('Temperature (°C)')
ax.set_title('Temperature Prediction using Locally Weighted Regression')
ax.legend()
ax.grid(True)

# Display the plot
st.pyplot(fig)

# Display some metrics
st.subheader("Data Insights")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Average Temperature", f"{np.mean(y):.1f}°C")
with col2:
    st.metric("Min Temperature", f"{np.min(y):.1f}°C")
with col3:
    st.metric("Max Temperature", f"{np.max(y):.1f}°C")

# Technical explanation
st.subheader("How it works")
st.write("""
Locally Weighted Regression (LWR) makes predictions by:
1. Giving more weight to nearby points using a Gaussian kernel
2. Fitting a local linear regression at each prediction point
3. The bandwidth parameter (τ) controls how much influence distant points have:
   - Larger τ = smoother predictions (more bias, less variance)
   - Smaller τ = more flexible predictions (less bias, more variance)
""")

# Show raw data
if st.checkbox("Show raw data"):
    st.write(df)
