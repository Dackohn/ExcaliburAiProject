from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Функция для отображения сентимента
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

# Применяем предобработку текста
data['text'] = data['text'].apply(preprocess_text)

# Применяем map_sentiment_and_reason для создания целевой переменной y
data['y'] = data.apply(map_sentiment_and_reason, axis=1)

# Векторизация текста
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(data['text'])

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, data['y'], test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)

# Обучение модели
model = LogisticRegression()
model.fit(X_train, y_train)

# Оценка модели
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
