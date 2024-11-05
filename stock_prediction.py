import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse

def fetch_stock_data(symbol, start_date, end_date):
    """Fetch stock data from Yahoo Finance"""
    stock = yf.Ticker(symbol)
    df = stock.history(start=start_date, end=end_date)
    return df

def prepare_data(df, lookback=30):
    """Prepare data for linear regression"""
    # Create features (previous n days' prices)
    X = []
    y = []
    
    # Get the closing prices
    prices = df['Close'].values
    
    for i in range(len(df) - lookback):
        X.append(prices[i:i+lookback])
        y.append(prices[i+lookback])
    
    return np.array(X), np.array(y)

def train_model(X, y):
    """Train the linear regression model"""
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, X_test, y_test, y_pred, mse, r2

def plot_results(y_test, y_pred, symbol):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title(f'Actual vs Predicted Stock Prices for {symbol}')
    plt.tight_layout()
    plt.savefig(f'prediction_results_{symbol}.png')
    plt.close()

def get_next_tuesday():
    """Get the date of next Tuesday"""
    today = datetime.now()
    days_until_tuesday = (1 - today.weekday()) % 7  # 1 = Tuesday
    if days_until_tuesday == 0:  # If today is Tuesday, get next Tuesday
        days_until_tuesday = 7
    next_tuesday = today + timedelta(days=days_until_tuesday)
    return next_tuesday.strftime('%Y-%m-%d')

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Stock Price Prediction using Linear Regression')
    parser.add_argument('--symbol', type=str, default='AAPL',
                      help='Stock symbol (default: AAPL)')
    parser.add_argument('--days', type=int, default=365*2,
                      help='Number of days of historical data to use (default: 730)')
    parser.add_argument('--lookback', type=int, default=30,
                      help='Number of days to look back for prediction (default: 30)')
    
    args = parser.parse_args()
    
    # Set parameters
    symbol = args.symbol
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    lookback = args.lookback
    
    # Fetch data
    print(f"Fetching stock data for {symbol}...")
    df = fetch_stock_data(symbol, start_date, end_date)
    
    if df.empty:
        print(f"Error: No data found for symbol {symbol}")
        return
    
    # Prepare data
    print("Preparing data...")
    X, y = prepare_data(df, lookback)
    
    # Train model and get predictions
    print("Training model...")
    model, X_test, y_test, y_pred, mse, r2 = train_model(X, y)
    
    # Print metrics
    print("\nModel Performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared Score: {r2:.2f}")
    
    # Plot results
    print("\nPlotting results...")
    plot_results(y_test, y_pred, symbol)
    print(f"Results have been saved to 'prediction_results_{symbol}.png'")
    
    # Make a prediction for next Tuesday
    last_prices = df['Close'].values[-lookback:]
    next_tuesday_pred = model.predict([last_prices])[0]
    next_tuesday_date = get_next_tuesday()
    
    print(f"\nPrediction for {symbol} on next Tuesday ({next_tuesday_date}):")
    print(f"Predicted price: ${next_tuesday_pred:.2f}")
    
    # Get current price for comparison
    current_price = df['Close'].iloc[-1]
    price_change = next_tuesday_pred - current_price
    price_change_percent = (price_change / current_price) * 100
    
    print(f"Current price: ${current_price:.2f}")
    print(f"Predicted change: ${price_change:.2f} ({price_change_percent:.2f}%)")

if __name__ == "__main__":
    main()
