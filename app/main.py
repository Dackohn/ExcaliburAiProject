import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import re
import seaborn as sns

#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('stopwords')

data = pd.read_csv('app/Tweets.csv')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
	text = re.sub(r'[^a-zA-Z]', ' ', text)
	text = text.lower()
	text = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words] #eliminare stopwords + lemantizare
	text = ' '.join(text) # reconstruieste textul
	return text

data['text'] = data['text'].apply(preprocess_text)
data['text'].head()

print(data['text'].head())
