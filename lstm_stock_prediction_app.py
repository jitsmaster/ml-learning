import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers.legacy import Adam  # Using legacy optimizer for M1/M2 Macs
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tensorflow as tf

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def calculate_technical_indicators(df):
    """Calculate technical indicators for better prediction"""
    # Moving averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Trading Volume
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    
    # Volatility
    df['Volatility'] = df['Close'].rolling(window=20).std()
    
    # Fill NaN values
    df.fillna(method='bfill', inplace=True)
    return df

def fetch_stock_data(symbol, days=365*2):
    """Fetch stock data from Yahoo Finance"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            raise ValueError(f"No data found for symbol {symbol}")
            
        df = calculate_technical_indicators(df)
        return df
    except Exception as e:
        raise Exception(f"Error fetching data: {str(e)}")

def prepare_data(df, lookback=60):
    """Prepare data for LSTM model"""
    # Features we'll use for prediction
    features = ['Close', 'Volume', 'MA5', 'MA20', 'RSI', 'Volume_MA5', 'Volatility']
    
    # Create feature matrix
    data = df[features].values
    
    # Scale the features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(len(df) - lookback):
        X.append(scaled_data[i:i+lookback])
        y.append(scaled_data[i+lookback, 0])  # 0 index for Close price
    
    return np.array(X), np.array(y), scaler, features

def create_lstm_model(lookback, n_features):
    """Create and compile LSTM model"""
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(lookback, n_features)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def get_next_tuesday():
    """Get the date of next Tuesday"""
    today = datetime.now()
    days_until_tuesday = (1 - today.weekday()) % 7
    if days_until_tuesday == 0:
        days_until_tuesday = 7
    next_tuesday = today + timedelta(days=days_until_tuesday)
    return next_tuesday.strftime('%Y-%m-%d')

def predict_stock_price(symbol, days, lookback):
    """Main prediction function"""
    try:
        # Fetch and prepare data
        df = fetch_stock_data(symbol, int(days))
        X, y, scaler, features = prepare_data(df, int(lookback))
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Create and train model
        model = create_lstm_model(lookback, X.shape[2])
        
        with st.spinner('Training model... This may take a few moments.'):
            history = model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_split=0.1,
                verbose=0
            )
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Prepare data for next Tuesday prediction
        last_sequence = X[-1:]
        next_tuesday_pred = model.predict(last_sequence)[0][0]
        
        # Create dummy array with same shape as original feature set
        pred_feature_array = np.zeros((1, len(features)))
        pred_feature_array[0, 0] = next_tuesday_pred  # Set the Close price prediction
        
        # Inverse transform predictions
        next_tuesday_pred = scaler.inverse_transform(pred_feature_array)[0, 0]
        current_price = df['Close'].iloc[-1]
        
        # Calculate error metrics
        # Ensure y_test and y_pred are the same shape
        y_pred_flat = y_pred.flatten()
        mse = np.mean((y_test - y_pred_flat) ** 2)
        rmse = np.sqrt(mse)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot 1: Training History
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Training History')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot 2: Actual vs Predicted
        # Create dummy arrays for inverse transformation
        test_feature_array = np.zeros((len(y_test), len(features)))
        test_feature_array[:, 0] = y_test
        actual_prices = scaler.inverse_transform(test_feature_array)[:, 0]
        
        pred_feature_array = np.zeros((len(y_pred), len(features)))
        pred_feature_array[:, 0] = y_pred.flatten()
        predicted_prices = scaler.inverse_transform(pred_feature_array)[:, 0]
        
        ax2.plot(actual_prices, label='Actual Prices')
        ax2.plot(predicted_prices, label='Predicted Prices')
        ax2.set_title('Actual vs Predicted Stock Prices')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Price')
        ax2.legend()
        
        plt.tight_layout()
        
        return {
            'current_price': float(current_price),  # Convert to Python float
            'next_tuesday_pred': float(next_tuesday_pred),  # Convert to Python float
            'metrics': {
                'rmse': float(rmse),  # Convert to Python float
                'mse': float(mse)  # Convert to Python float
            }
        }, fig, None
        
    except Exception as e:
        return None, None, str(e)

# Streamlit UI
st.set_page_config(page_title="Advanced Stock Price Prediction", layout="wide")

st.title("Advanced Stock Price Prediction using LSTM")
st.write("""
This app uses a Long Short-Term Memory (LSTM) neural network to predict stock prices.
LSTMs are particularly well-suited for time series prediction as they can capture long-term patterns and dependencies in the data.
""")

# Input form
col1, col2, col3 = st.columns(3)

with col1:
    symbol = st.text_input("Stock Symbol", value="AAPL", help="Enter stock symbol (e.g., AAPL, GOOGL, MSFT)")

with col2:
    days = st.slider("Historical Days", min_value=365, max_value=1825, value=730, 
                     help="Number of historical days to use for training")

with col3:
    lookback = st.slider("Lookback Days", min_value=30, max_value=120, value=60,
                        help="Number of previous days to use for prediction")

if st.button("Predict"):
    with st.spinner("Fetching data and making predictions..."):
        results, fig, error = predict_stock_price(symbol, days, lookback)
        
        if error:
            st.error(f"Error: {error}")
        elif results:
            # Display results
            next_tuesday = get_next_tuesday()
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"${results['current_price']:.2f}")
            with col2:
                st.metric("Predicted Price", f"${results['next_tuesday_pred']:.2f}")
            with col3:
                change = results['next_tuesday_pred'] - results['current_price']
                change_percent = (change / results['current_price']) * 100
                st.metric("Expected Change", 
                         f"${change:.2f} ({change_percent:.2f}%)",
                         delta=f"{change_percent:.2f}%")
            
            # Model performance
            st.subheader("Model Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Root Mean Squared Error", f"{results['metrics']['rmse']:.3f}")
            with col2:
                st.metric("Mean Squared Error", f"{results['metrics']['mse']:.2f}")
            
            # Plot
            st.subheader("Model Analysis")
            st.pyplot(fig)
            
            # Additional information
            st.info(f"Prediction is for next Tuesday ({next_tuesday})")
            
            st.subheader("Model Details")
            st.write("""
            This prediction uses several technical indicators:
            - 5-day and 20-day Moving Averages
            - Relative Strength Index (RSI)
            - Volume Moving Average
            - Price Volatility
            
            The LSTM model architecture includes:
            - Two LSTM layers with dropout for regularization
            - Two Dense layers for final prediction
            - Adam optimizer with learning rate of 0.001
            """)

# Example section
st.sidebar.header("Example Stocks")
st.sidebar.write("Try these symbols:")
example_stocks = {
    "AAPL": "Apple Inc.",
    "GOOGL": "Alphabet Inc.",
    "MSFT": "Microsoft Corporation",
    "AMZN": "Amazon.com Inc.",
    "TSLA": "Tesla Inc.",
    "NVDA": "NVIDIA Corporation",
    "META": "Meta Platforms Inc."
}
for symbol, name in example_stocks.items():
    st.sidebar.write(f"• {symbol}: {name}")

st.sidebar.markdown("""
---
### Why LSTM?
Long Short-Term Memory (LSTM) networks are more suitable for stock prediction than linear regression because they can:
1. Capture non-linear patterns in data
2. Remember important patterns over long sequences
3. Handle time-dependent patterns
4. Learn complex market behaviors
""")
