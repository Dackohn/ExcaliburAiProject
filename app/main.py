from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
from gensim.models import Word2Vec

# Function to map sentiment and reason
def map_sentiment_and_reason(row):
    if row['airline_sentiment'] == 'positive':
        return 12
    elif row['airline_sentiment'] == 'neutral':
        return 11
    elif row['airline_sentiment'] == 'negative':
        reason_mapping = {
            'Customer Service Issue': 0,
            'Late Flight': 1,
            "Can't Tell": 2,
            'Cancelled Flight': 3,
            'Lost Luggage': 4,
            'Bad Flight': 5,
            'Flight Booking Problems': 6,
            'Flight Attendant Complaints': 7,
            'longlines': 8,
            'Damaged Luggage': 9
        }
        return reason_mapping.get(row['negativereason'], 10)  # Default to 10 if no specific reason
    return None  # Handle missing cases

# Load data
data = pd.read_csv('Tweets.csv')

# Initialize stop words and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = nltk.WordNetLemmatizer()

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert to lower case
    text = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]  # Lemmatization and stopword removal
    text = ' '.join(text)  # Return the text as a string
    return text

# Apply text preprocessing
data['text'] = data['text'].apply(preprocess_text)

# Apply map_sentiment_and_reason to create the target variable y
data['y'] = data.apply(map_sentiment_and_reason, axis=1)

# Tokenize text for Word2Vec
tokenized_text = [text.split() for text in data['text']]

# Train Word2Vec model
word2vec_model = Word2Vec(sentences=tokenized_text, vector_size=100, window=5, min_count=1, workers=4)
word2vec_model.save("word2vec.model")

# Function to convert text to average Word2Vec vector
def get_average_word2vec(tokens, model, vector_size=100):
    vector = np.zeros(vector_size)
    num_tokens = 0
    for word in tokens:
        if word in model.wv:
            vector += model.wv[word]
            num_tokens += 1
    if num_tokens > 0:
        vector /= num_tokens
    return vector

# Apply Word2Vec to convert each document into a feature vector
X_word2vec = np.array([get_average_word2vec(text.split(), word2vec_model) for text in data['text']])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_word2vec, data['y'], test_size=0.2, random_state=42)

# Models for training
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# Train and evaluate models
for model_name, model in models.items():
    print(f"Training model: {model_name}")
    
    # GridSearchCV for Random Forest and Logistic Regression
    if model_name == "Random Forest":
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        print(f"Best parameters for Random Forest: {grid_search.best_params_}")
        model = grid_search.best_estimator_
        
    elif model_name == "Logistic Regression":
        param_grid = {
            'C': [0.1, 1, 10],
            'solver': ['liblinear', 'saga']
        }
        grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        print(f"Best parameters for Logistic Regression: {grid_search.best_params_}")
        model = grid_search.best_estimator_

    # Train the model
    model.fit(X_train, y_train)
    
    # Prediction
    y_pred = model.predict(X_test)
    
    # Model evaluation
    print(f"Results for {model_name}:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print("=" * 60)
