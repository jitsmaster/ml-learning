import numpy as np
from collections import Counter
import re

class TextEmotionSVM:
    def __init__(self, C=1.0, kernel='rbf', gamma=0.5, max_iter=1000):
        self.C = C  # Regularization parameter
        self.kernel = kernel
        self.gamma = gamma  # RBF kernel parameter
        self.max_iter = max_iter
        self.vocab = None
        self.support_vectors = None
        self.alphas = None
        self.b = None
        self.support_vector_labels = None
        
        # Sentiment indicators
        self.positive_words = {'great', 'wonderful', 'happy', 'excited', 'amazing', 'perfect', 'good', 'love', 'excellent'}
        self.negative_words = {'terrible', 'worst', 'angry', 'hate', 'devastated', 'disappointing', 'bad', 'awful', 'horrible'}
        self.neutral_words = {'okay', 'normal', 'average', 'fine', 'alright', 'regular', 'usual', 'standard'}
        
    def _preprocess_text(self, text):
        """Convert text to lowercase and split into tokens"""
        text = text.lower()
        # Remove special characters and split into tokens
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def _create_vocabulary(self, texts):
        """Create vocabulary from all training texts"""
        all_tokens = []
        for text in texts:
            tokens = self._preprocess_text(text)
            all_tokens.extend(tokens)
        # Keep words that appear at least once
        word_counts = Counter(all_tokens)
        self.vocab = {word: idx for idx, (word, count) 
                     in enumerate(word_counts.items()) if count >= 1}
        
    def _get_sentiment_features(self, tokens):
        """Extract sentiment-based features from tokens"""
        pos_count = sum(1 for token in tokens if token in self.positive_words)
        neg_count = sum(1 for token in tokens if token in self.negative_words)
        neu_count = sum(1 for token in tokens if token in self.neutral_words)
        
        total = len(tokens) + 1e-10  # Avoid division by zero
        return np.array([
            pos_count / total,  # Positive ratio
            neg_count / total,  # Negative ratio
            neu_count / total,  # Neutral ratio
            pos_count - neg_count,  # Sentiment difference
            1 if pos_count > neg_count else -1 if neg_count > pos_count else 0  # Overall polarity
        ])
    
    def _text_to_vector(self, text):
        """Convert text to feature vector combining TF-IDF and sentiment features"""
        if self.vocab is None:
            raise ValueError("Vocabulary not created. Call fit() first.")
            
        tokens = self._preprocess_text(text)
        vocab_size = len(self.vocab)
        
        # TF-IDF like features
        tf = np.zeros(vocab_size)
        for token in tokens:
            if token in self.vocab:
                idx = self.vocab[token]
                if idx < vocab_size:
                    tf[idx] += 1
                    
        # Normalize TF
        token_sum = np.sum(tf)
        if token_sum > 0:
            tf = tf / token_sum
            
        # Get sentiment features
        sentiment_features = self._get_sentiment_features(tokens)
        
        # Combine all features
        return np.concatenate([tf, sentiment_features])
    
    def _rbf_kernel(self, x1, x2):
        """Compute RBF kernel between two vectors"""
        diff = x1 - x2
        return np.exp(-self.gamma * np.dot(diff, diff))
    
    def _kernel_matrix(self, X1, X2=None):
        """Compute kernel matrix between two sets of vectors"""
        if X2 is None:
            X2 = X1
            
        K = np.zeros((X1.shape[0], X2.shape[0]))
        for i in range(X1.shape[0]):
            for j in range(X2.shape[0]):
                K[i,j] = self._rbf_kernel(X1[i], X2[j])
        return K
    
    def fit(self, texts, labels):
        """Train the SVM model"""
        # Create vocabulary and convert texts to vectors
        self._create_vocabulary(texts)
        X = np.array([self._text_to_vector(text) for text in texts])
        y = np.array(labels, dtype=float)
        
        # Normalize labels to [-1, 1] range for better SVM performance
        self.y_min = min(y)
        self.y_max = max(y)
        y = 2 * (y - self.y_min) / (self.y_max - self.y_min) - 1
        
        n_samples = X.shape[0]
        
        # Compute kernel matrix
        K = self._kernel_matrix(X)
        
        # Initialize variables for SMO algorithm
        self.alphas = np.zeros(n_samples)
        self.b = 0
        
        # Simple SMO algorithm
        changed = True
        iter_count = 0
        
        while changed and iter_count < self.max_iter:
            changed = False
            iter_count += 1
            
            for i in range(n_samples):
                Ei = np.sum(self.alphas * y * K[i]) + self.b - y[i]
                
                if ((y[i] * Ei < -0.01 and self.alphas[i] < self.C) or
                    (y[i] * Ei > 0.01 and self.alphas[i] > 0)):
                    
                    # Select second alpha randomly
                    j = i
                    while j == i:
                        j = np.random.randint(n_samples)
                        
                    Ej = np.sum(self.alphas * y * K[j]) + self.b - y[j]
                    
                    old_ai = self.alphas[i]
                    old_aj = self.alphas[j]
                    
                    if y[i] != y[j]:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0, self.alphas[i] + self.alphas[j] - self.C)
                        H = min(self.C, self.alphas[i] + self.alphas[j])
                        
                    if L == H:
                        continue
                        
                    eta = 2 * K[i,j] - K[i,i] - K[j,j]
                    if eta >= 0:
                        continue
                        
                    self.alphas[j] = old_aj - y[j] * (Ei - Ej) / eta
                    self.alphas[j] = min(H, max(L, self.alphas[j]))
                    
                    if abs(self.alphas[j] - old_aj) < 1e-5:
                        continue
                        
                    self.alphas[i] = old_ai + y[i] * y[j] * (old_aj - self.alphas[j])
                    
                    b1 = self.b - Ei - y[i] * (self.alphas[i] - old_ai) * K[i,i] \
                         - y[j] * (self.alphas[j] - old_aj) * K[i,j]
                    b2 = self.b - Ej - y[i] * (self.alphas[i] - old_ai) * K[i,j] \
                         - y[j] * (self.alphas[j] - old_aj) * K[j,j]
                    
                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
                        
                    changed = True
        
        # Store support vectors
        sv_indices = self.alphas > 1e-5
        self.support_vectors = X[sv_indices]
        self.support_vector_labels = y[sv_indices]
        self.alphas = self.alphas[sv_indices]
        
        return self
    
    def predict(self, texts):
        """Predict emotional intensity (1-5) for new texts"""
        if self.support_vectors is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        X = np.array([self._text_to_vector(text) for text in texts])
        K = self._kernel_matrix(X, self.support_vectors)
        
        # Get raw predictions
        predictions = np.zeros(len(texts))
        for i in range(len(texts)):
            predictions[i] = np.sum(self.alphas * self.support_vector_labels * K[i]) + self.b
        
        # Scale predictions back to original range
        scaled_predictions = (predictions + 1) / 2 * (self.y_max - self.y_min) + self.y_min
        
        return np.clip(np.round(scaled_predictions), 1, 5).astype(int)

# Example usage
if __name__ == "__main__":
    # Example training data with more diverse examples
    train_texts = [
        "I am feeling absolutely wonderful today!",
        "This is the worst day ever.",
        "The weather is okay, nothing special.",
        "I'm so excited about the upcoming vacation!",
        "I'm really angry about what happened.",
        "Just another normal day at work.",
        "This makes me incredibly happy!",
        "I'm devastated by the news.",
        "Everything is going smoothly.",
        "I can't believe how amazing this is!",
        "I love this so much!",
        "This is absolutely horrible.",
        "It's pretty average overall.",
        "I'm feeling great about everything!",
        "This is disappointing and frustrating."
    ]
    
    # Labels: 1 (very negative) to 5 (very positive)
    train_labels = [5, 1, 3, 5, 1, 3, 5, 1, 3, 5, 5, 1, 3, 5, 1]
    
    # Create and train the model
    model = TextEmotionSVM(C=1.0, gamma=0.5)
    model.fit(train_texts, train_labels)
    
    # Test the model
    test_texts = [
        "I'm having such a great time!",
        "This is terrible, I hate it.",
        "It's an average day today.",
        "This is somewhat disappointing.",
        "Everything is absolutely perfect!",
        "I love this product so much!",
        "This is the worst experience ever.",
        "Things are going okay I guess.",
        "I'm really excited about this!",
        "This makes me so angry."
    ]
    
    predictions = model.predict(test_texts)
    
    print("\nPredictions:")
    for text, pred in zip(test_texts, predictions):
        print(f"Text: '{text}'")
        print(f"Emotional Level (1-5): {pred}\n")
