import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

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
        return reason_mapping.get(row['negativereason'], 10)
    return None

# Load data
data = pd.read_csv('Tweets.csv')

# Initialize stop words and lemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = nltk.WordNetLemmatizer()

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(text)

# Apply text preprocessing
data['text'] = data['text'].apply(preprocess_text)

# Apply map_sentiment_and_reason to create the target variable y
data['y'] = data.apply(map_sentiment_and_reason, axis=1)

# Tokenize text for Word2Vec
tokenized_text = [text.split() for text in data['text']]

# Train Word2Vec model with reduced vector size
word2vec_model = Word2Vec(sentences=tokenized_text, vector_size=30, window=3, min_count=1, workers=2)
word2vec_model.save("word2vec.model")

# Function to convert text to average Word2Vec vector
def get_average_word2vec(tokens, model, vector_size=30):
    vector = np.zeros(vector_size)
    num_tokens = 0
    for word in tokens:
        if word in model.wv:
            vector += model.wv[word]
            num_tokens += 1
    return vector / num_tokens if num_tokens > 0 else vector

# Precompute Word2Vec vectors
X_word2vec = np.array([get_average_word2vec(text.split(), word2vec_model) for text in data['text']])

# Encode sentiment labels
le = LabelEncoder()
data['y'] = le.fit_transform(data['airline_sentiment'])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_word2vec, data['y'], test_size=0.2, random_state=42)

# Models for training
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(random_state=42),
    "Neural Network": Sequential()
}

# Neural Network model
models["Neural Network"].add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
models["Neural Network"].add(Dropout(0.2))
models["Neural Network"].add(Dense(64, activation='relu'))
models["Neural Network"].add(Dense(32, activation='relu'))
models["Neural Network"].add(Dense(3, activation='softmax'))  # 3 classes: positive, negative, neutral
models["Neural Network"].compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Hyperparameter tuning configuration for Random Forest and SVM
param_grids = {
    "Random Forest": {
        'n_estimators': [50, 100],
        'max_depth': [10, None],
    },
    "SVM": {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }
}

# Store the results in a dictionary for comparison
results = {}

# Train and evaluate models
for model_name, model in models.items():
    print(f"Training model: {model_name}")
    if model_name != "Neural Network":  # Random Forest and SVM
        param_grid = param_grids[model_name]
        random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=5, cv=2, n_jobs=-1, verbose=1, random_state=42)
        random_search.fit(X_train, y_train)

        print(f"Best parameters for {model_name}: {random_search.best_params_}")
        model = random_search.best_estimator_

        # Prediction
        y_pred = model.predict(X_test)
    else:  # Neural Network
        model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))
        y_pred = model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)

    # Store evaluation results for comparison
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    results[model_name] = {
        'accuracy': accuracy,
        'classification_report': class_report
    }

    print(f"Accuracy for {model_name}: {accuracy}")
    print(f"Classification report for {model_name}: {class_report}")
    print("=" * 60)

# Find the best model based on accuracy
best_model_name = max(results, key=lambda model_name: results[model_name]['accuracy'])
best_model = results[best_model_name]

# Output the best model's result
print("\nBest Model:")
print(f"Model: {best_model_name}")
print(f"Accuracy: {best_model['accuracy']}")
print(f"Classification Report:\n{best_model['classification_report']}")
