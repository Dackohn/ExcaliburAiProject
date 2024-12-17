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
from imblearn.under_sampling import RandomUnderSampler


#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('stopwords')

def map_sentiment_and_reason(row):
    if row == 'positive':
        return 12
    elif row == 'neutral':
        return 11
    elif row == 'negative':
        return 10 
    return None  

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
# class balancing under sampling   
X = data.drop(['airline_sentiment'], axis=1)
y = data['airline_sentiment']

rus = RandomUnderSampler(sampling_strategy="not minority") # String
X_res, y_res = rus.fit_resample(X, y)

ax = y_res.value_counts().plot.pie(autopct='%1.1f%%',figsize=(8, 8))
_ = ax.set_title("Under-sampling")

#Vectorizare text
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(X_res['text'])

# Aplică funcția pe fiecare rând
y = y_res.apply(map_sentiment_and_reason)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)

