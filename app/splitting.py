from sklearn.model_selection import train_test_split
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

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

# Загрузка данных
data = pd.read_csv('Tweets.csv')

# Инициализация стоп-слов и лемматизатора
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Функция предобработки текста
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Удаление всех символов, кроме букв
    text = text.lower()  # Преобразование в нижний регистр
    text = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]  # Лемматизация и удаление стоп-слов
    text = ' '.join(text)  # Возвращаем текст обратно в строку
    return text

# Применяем предобработку к данным
data['text'] = data['text'].apply(preprocess_text)
print(data['text'].head())  # Печать первых 5 строк

# Разделяем данные на признаки (X) и целевую переменную (y)
X = data['text']
y = data['airline_sentiment']

# Разделяем данные на обучающую (70%) и временную (30%) выборки
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

# Разделяем временную выборку на валидационную (50%) и тестовую (50%) выборки
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Печать размеров наборов данных
print("Training data size:", X_train.shape[0])
print("Validation data size:", X_val.shape[0])
print("Test data size:", X_test.shape[0])
