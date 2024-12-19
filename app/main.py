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
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
import joblib

#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('stopwords')
def transform_sentiment_to_number(row):
    if row == 'positive':
        return 2
    elif row == 'neutral':
        return 1
    elif row == 'negative':
        return 0
    return None

def map_sentiment(row):
    if row == 'positive':
        return 'positive'
    elif row == 'neutral':
        return 'neutral'
    elif row == 'negative':
        return 'negative'
    elif row == 'Positive':
        return 'positive'
    elif row == 'Neutral':
        return 'neutral'
    elif row == 'Negative':
        return 'negative'
    elif row == 'Irrelevant':
        return 'neutral'
    return None

def preprocess_text(text):
    # Elimină caracterele non-alfabetice
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Transformă textul în litere mici
    text = text.lower()
    # Tokenizare și eliminarea stopwords + lematizare
    text = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    text = ' '.join(text)
    return text

# Citirea dataseturilor și extragerea textului și a sentimentelor
data_tweets = pd.read_csv('https://raw.githubusercontent.com/Dackohn/ExcaliburAiProject/refs/heads/main/datasets/Tweets.csv')
data_tweets=data_tweets[['text','airline_sentiment']]

data_sentiment = pd.read_csv('https://raw.githubusercontent.com/Dackohn/ExcaliburAiProject/refs/heads/main/datasets/sentimentdataset.csv')
data_sentiment=data_sentiment[['Text','Sentiment']]

data_imdb = pd.read_csv('https://raw.githubusercontent.com/Dackohn/ExcaliburAiProject/refs/heads/main/datasets/IMDB-Dataset.csv')

data_text2 = pd.read_csv('https://raw.githubusercontent.com/Dackohn/ExcaliburAiProject/refs/heads/main/datasets/train.csv', encoding='unicode_escape')
data_text2 = data_text2[['text','sentiment']]

data_text1 = pd.read_csv('https://raw.githubusercontent.com/Dackohn/ExcaliburAiProject/refs/heads/main/datasets/test.csv', encoding='unicode_escape')
data_text1 = data_text1[['text','sentiment']]

data_titter = pd.read_csv('https://raw.githubusercontent.com/Dackohn/ExcaliburAiProject/refs/heads/main/datasets/twitter_training.csv', encoding='unicode_escape')
nume_col = ['Coloana1', 'Coloana2','sentiment', 'text']
data_titter.columns = nume_col
data_titter = data_titter[['text','sentiment']]

# Redenumirea coloanelor pentru consistență
data_imdb = data_imdb.rename(columns={
    'review': 'text',
    'sentiment': 'sentiment'
})

data_tweets = data_tweets.rename(columns={
    'review': 'text',
    'airline_sentiment': 'sentiment'
})

data_sentiment = data_sentiment.rename(columns={
    'Text': 'text',
    'Sentiment': 'sentiment'
})

# Combinarea tuturor dataseturilor într-un singur DataFrame
data_combined = pd.concat([data_titter, data_imdb, data_sentiment, data_text1, data_text2, data_tweets], ignore_index=True)
# Normalizarea sentimentelor
data_combined['sentiment'] = data_combined['sentiment'].apply(map_sentiment)
# Eliminarea valorilor lipsă
data_combined = data_combined.dropna(subset=['sentiment'])
data_combined = data_combined.dropna(subset=['text'])

# Balansare dataset
X = data_combined['text']
y = data_combined['sentiment']

rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)
X_res, y_res = rus.fit_resample(X.values.reshape(-1, 1), y)

data_combined_balanced = pd.DataFrame({
    'text': X_res.flatten(),
    'sentiment': y_res
})

# Actualizați 'data_combined' cu dataset-ul echilibrat
data_combined = data_combined_balanced

# Definirea stopwords și lematizatorului
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocesarea textului
data_combined['text'] = data_combined['text'].apply(preprocess_text)

X_res = data_combined['text']
y_res = data_combined['sentiment']

# Vectorizare text
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(X_res)

# Transformarea sentimentelor în numere
y = y_res.apply(transform_sentiment_to_number)

# Implementare K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = LogisticRegression(max_iter=1000)
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
print("K-Fold Cross Validation Accuracy Scores:", scores)
print("Mean Accuracy:", scores.mean())

model.fit(X, y)
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
