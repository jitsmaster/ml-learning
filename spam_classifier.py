import numpy as np
import pandas as pd
from typing import Tuple, List
import re
from sklearn.preprocessing import StandardScaler

class LogisticRegression:
    def __init__(self, learning_rate: float = 0.01, num_iterations: int = 1000):
        """
        Initialize Logistic Regression model
        
        Args:
            learning_rate: Step size for gradient descent
            num_iterations: Number of training iterations
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.scaler = StandardScaler()
        
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Compute sigmoid activation function
        
        Args:
            z: Input values
            
        Returns:
            Sigmoid of input values
        """
        # Clip values to avoid overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the logistic regression model using gradient descent
        
        Args:
            X: Training features (n_samples, n_features)
            y: Target values (n_samples,)
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        n_samples, n_features = X_scaled.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.num_iterations):
            # Forward pass
            linear_pred = np.dot(X_scaled, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X_scaled.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            X: Features to predict on
            threshold: Classification threshold
            
        Returns:
            Binary predictions (0 or 1)
        """
        X_scaled = self.scaler.transform(X)
        linear_pred = np.dot(X_scaled, self.weights) + self.bias
        probabilities = self.sigmoid(linear_pred)
        return (probabilities >= threshold).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability predictions
        
        Args:
            X: Features to predict on
            
        Returns:
            Probability predictions
        """
        X_scaled = self.scaler.transform(X)
        linear_pred = np.dot(X_scaled, self.weights) + self.bias
        return self.sigmoid(linear_pred)

def extract_email_features(email_text: str) -> List[float]:
    """
    Extract features from email text
    
    Args:
        email_text: Raw email content
        
    Returns:
        List of extracted features
    """
    features = []
    
    # Text length features
    features.append(len(email_text))  # Total length
    features.append(len(email_text.split()))  # Word count
    
    # Character-based features
    features.append(len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', email_text)))  # Special chars
    features.append(len(re.findall(r'\d', email_text)))  # Numbers
    features.append(sum(1 for c in email_text if c.isupper()) / max(len(email_text), 1))  # Ratio of uppercase
    
    # Word-based features
    lowercase_text = email_text.lower()
    words = lowercase_text.split()
    
    # Money-related features
    money_words = len(re.findall(r'\b(money|cash|dollar|price|offer|deal|win|won|prize)\b', lowercase_text))
    features.append(money_words)
    features.append(len(re.findall(r'[\$£€]', email_text)))  # Currency symbols
    
    # Urgency-related features
    urgent_words = len(re.findall(r'\b(urgent|important|action|now|limited|hurry|fast|quick|expires|today)\b', lowercase_text))
    features.append(urgent_words)
    
    # Spam-indicative features
    features.append(len(re.findall(r'\b(free|bonus|discount|offer|special|exclusive|guaranteed|amazing)\b', lowercase_text)))
    features.append(email_text.count('!') / max(len(email_text.split()), 1))  # Exclamation marks per word
    
    # Link and number features
    features.append(len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', email_text)))
    features.append(len(re.findall(r'\d+(?:\.\d+)?%', email_text)))  # Percentage numbers
    
    # Formatting features
    features.append(len(re.findall(r'\b[A-Z]{2,}\b', email_text)))  # Words in ALL CAPS
    
    return features

# Example usage
if __name__ == "__main__":
    # Generate synthetic email data
    np.random.seed(42)
    
    # Create synthetic spam and non-spam emails
    spam_emails = [
        "URGENT: You've won $1,000,000! Claim NOW!",
        "Make MONEY Fast! Limited Time Offer!!!",
        "IMPORTANT: Your Account needs verification $$$",
        "Special deal! 90% OFF - Act Now!!!",
        "Congratulations! You're our lucky WINNER!",
        "FREE VIAGRA! Best prices guaranteed!!!",
        "Make $5000 weekly from home! Easy money!",
        "EXCLUSIVE: Limited time 75% discount!!!",
        "Your PayPal account needs verification ASAP",
        "WIN WIN WIN! Lottery results inside!!!"
    ]
    
    normal_emails = [
        "Meeting scheduled for tomorrow at 2 PM",
        "Please review the attached document",
        "Thank you for your recent purchase",
        "Project update: Phase 1 complete",
        "Reminder: Team lunch next week",
        "Could you send me the quarterly report?",
        "The presentation went well today",
        "Happy birthday! Hope you have a great day",
        "Your order has been shipped",
        "Notes from yesterday's meeting attached"
    ]
    
    # Extract features and create training data
    X_train = []
    y_train = []
    
    # Add spam examples
    for email in spam_emails:
        X_train.append(extract_email_features(email))
        y_train.append(1)  # 1 for spam
        
    # Add normal examples
    for email in normal_emails:
        X_train.append(extract_email_features(email))
        y_train.append(0)  # 0 for non-spam
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Train the model
    model = LogisticRegression(learning_rate=0.01, num_iterations=2000)
    model.fit(X_train, y_train)
    
    # Test predictions
    test_emails = [
        "Hello, the meeting is at 3 PM tomorrow.",
        "CONGRATULATIONS! You've WON $5,000,000! CLAIM NOW!!!",
        "Please review the Q2 financial report",
        "URGENT: Make $1000 daily working from home!!!",
        "Your package has been delivered to the mailroom",
        "FREE iPhone 14! You're our 1,000,000th visitor!!!"
    ]
    
    print("Test Results:")
    print("-" * 50)
    
    for email in test_emails:
        features = extract_email_features(email)
        prediction = model.predict(np.array([features]))[0]
        probability = model.predict_proba(np.array([features]))[0]
        
        print("\nEmail:", email)
        print("Prediction:", "Spam" if prediction == 1 else "Not Spam")
        print(f"Probability of being spam: {probability:.2%}")
