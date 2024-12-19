from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
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
lemmatizer = nltk.WordNetLemmatizer()

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

# Векторизация текста с биграммами
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))  # Используем биграммы
X = vectorizer.fit_transform(data['text'])

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, data['y'], test_size=0.2, random_state=42)

# Модели для обучения
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# Обучение и оценка моделей
for model_name, model in models.items():
    print(f"Обучение модели: {model_name}")
    
    # Настроим гиперпараметры с помощью GridSearchCV (для Random Forest и Logistic Regression)
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

    # Обучаем модель
    model.fit(X_train, y_train)
    
    # Предсказание
    y_pred = model.predict(X_test)
    
    # Оценка модели
    print(f"Результаты {model_name}:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print("=" * 60)
