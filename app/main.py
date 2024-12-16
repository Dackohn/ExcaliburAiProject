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

data = pd.read_csv('Tweets.csv')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
	text = re.sub(r'[^a-zA-Z]', ' ', text)
	text = text.lower()
	text = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
	text = ' '.join(text)
	return text

data['text'] = data['text'].apply(preprocess_text)
data['text'].head()

print(data['text'].head())

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(data['text'])
X.shape
print(X)
