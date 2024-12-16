import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import re
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('stopwords')

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

def preprocess_text(text):
	text = re.sub(r'[^a-zA-Z]', ' ', text)
	text = text.lower()
	text = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
	text = ' '.join(text)
	return text


data = pd.read_csv('app\Tweets.csv')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

data['text'] = data['text'].apply(preprocess_text)
data['text'].head()

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(data['text'])

# Aplică funcția pe fiecare rând
y = data.apply(map_sentiment_and_reason, axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)

