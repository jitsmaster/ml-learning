import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def fetch_stock_data(symbol, days=365*2):
    """Fetch stock data from Yahoo Finance"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    stock = yf.Ticker(symbol)
    df = stock.history(start=start_date, end=end_date)
    return df

def prepare_data(df, lookback=30):
    """Prepare data for linear regression"""
    X = []
    y = []
    prices = df['Close'].values
    
    for i in range(len(df) - lookback):
        X.append(prices[i:i+lookback])
        y.append(prices[i+lookback])
    
    return np.array(X), np.array(y)

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
        # Fetch data
        df = fetch_stock_data(symbol, int(days))
        if df.empty:
            return None, None, None, "No data found for this symbol"
        
        # Prepare data
        X, y = prepare_data(df, int(lookback))
        
        # Split data and train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Predict next Tuesday
        last_prices = df['Close'].values[-int(lookback):]
        next_tuesday_pred = model.predict([last_prices])[0]
        current_price = df['Close'].iloc[-1]
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, y_pred, color='blue', alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Price')
        ax.set_ylabel('Predicted Price')
        ax.set_title(f'Actual vs Predicted Stock Prices for {symbol}')
        plt.tight_layout()
        
        return {
            'current_price': current_price,
            'next_tuesday_pred': next_tuesday_pred,
            'metrics': {
                'r2': r2,
                'mse': mse
            }
        }, fig, None
        
    except Exception as e:
        return None, None, str(e)

# Streamlit UI
st.set_page_config(page_title="Stock Price Prediction", layout="wide")

st.title("Stock Price Prediction for Next Tuesday")
st.write("Enter a stock symbol to predict its price for next Tuesday using linear regression.")

# Input form
col1, col2, col3 = st.columns(3)

with col1:
    symbol = st.text_input("Stock Symbol", value="AAPL", help="Enter stock symbol (e.g., AAPL, GOOGL, MSFT)")

with col2:
    days = st.slider("Historical Days", min_value=30, max_value=1825, value=365, 
                     help="Number of historical days to use for training")

with col3:
    lookback = st.slider("Lookback Days", min_value=5, max_value=60, value=30,
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
                st.metric("R-squared Score", f"{results['metrics']['r2']:.3f}")
            with col2:
                st.metric("Mean Squared Error", f"{results['metrics']['mse']:.2f}")
            
            # Plot
            st.subheader("Prediction Visualization")
            st.pyplot(fig)
            
            # Additional information
            st.info(f"Prediction is for next Tuesday ({next_tuesday})")

# Example section
st.sidebar.header("Example Stocks")
st.sidebar.write("Try these symbols:")
example_stocks = {
    "AAPL": "Apple Inc.",
    "GOOGL": "Alphabet Inc.",
    "MSFT": "Microsoft Corporation",
    "AMZN": "Amazon.com Inc.",
    "TSLA": "Tesla Inc."
}
for symbol, name in example_stocks.items():
    st.sidebar.write(f"• {symbol}: {name}")
