import numpy as np
import pandas as pd
import re
import nltk
import joblib
from nltk.corpus import stopwords
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.under_sampling import RandomUnderSampler

# NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
lemmatizer = nltk.WordNetLemmatizer()


def preprocess_text(text):
    if not isinstance(text, str):
        return None
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = [
        lemmatizer.lemmatize(word)
        for word in text.split() if word not in stop_words
    ]
    return ' '.join(text)


# Загрузка данных
data_urls = [
    'https://raw.githubusercontent.com/Dackohn/ExcaliburAiProject/refs/heads/'
    'main/datasets/Tweets.csv',
    'https://raw.githubusercontent.com/Dackohn/ExcaliburAiProject/refs/heads/'
    'main/datasets/sentimentdataset.csv',
    'https://raw.githubusercontent.com/Dackohn/ExcaliburAiProject/refs/heads/'
    'main/datasets/IMDB-Dataset.csv',
    'https://raw.githubusercontent.com/Dackohn/ExcaliburAiProject/refs/heads/'
    'main/datasets/train.csv',
    'https://raw.githubusercontent.com/Dackohn/ExcaliburAiProject/refs/heads/'
    'main/datasets/test.csv',
    'https://raw.githubusercontent.com/Dackohn/ExcaliburAiProject/refs/heads/'
    'main/datasets/twitter_training.csv'
]

dataframes = []
for url in data_urls:
    try:
        df = pd.read_csv(url, encoding='unicode_escape')
        dataframes.append(df)
    except Exception as e:
        print(f"Error loading {url}: {e}")

# Приведение данных
dataframes[0] = dataframes[0][['text', 'airline_sentiment']].rename(
    columns={'airline_sentiment': 'sentiment'}
)
dataframes[1] = dataframes[1][['Text', 'Sentiment']].rename(
    columns={'Text': 'text', 'Sentiment': 'sentiment'}
)
dataframes[2] = dataframes[2].rename(
    columns={'review': 'text', 'sentiment': 'sentiment'}
)
dataframes[3] = dataframes[3][['text', 'sentiment']]
dataframes[4] = dataframes[4][['text', 'sentiment']]
dataframes[5].columns = ['Coloana1', 'Coloana2', 'sentiment', 'text']
dataframes[5] = dataframes[5][['text', 'sentiment']]

# Объединение всех данных
data_combined = pd.concat(dataframes, ignore_index=True)
data_combined['text'] = data_combined['text'].astype(str).apply(
    preprocess_text
)
data_combined['sentiment'] = data_combined['sentiment'].map(
    {'positive': 2, 'neutral': 1, 'negative': 0}
)
data_combined.dropna(subset=['sentiment'], inplace=True)

# Балансировка
X = data_combined['text'].values.reshape(-1, 1)
y = data_combined['sentiment']
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X, y)

data_balanced = pd.DataFrame({'text': X_res.flatten(), 'sentiment': y_res})


# Использование TF-IDF для представления текста
def get_tfidf_features(texts):
    vectorizer = TfidfVectorizer(max_features=5000)
    return vectorizer.fit_transform(texts).toarray(), vectorizer


# Преобразование текста в TF-IDF признаки
X_tfidf, vectorizer = get_tfidf_features(data_balanced['text'])

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, data_balanced['sentiment'], test_size=0.2, random_state=42
)

# Нейронная сеть с улучшениями
model = Sequential([
    Dense(256, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.4),
    BatchNormalization(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])

# Использование оптимизатора Adam с уменьшением learning rate
optimizer = Adam(learning_rate=0.00005)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

early_stopping = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

print("Training Neural Network with EarlyStopping...")
model.fit(
    X_train, y_train, epochs=1, batch_size=32,
    validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping]
)

# Прогнозирование и оценка точности
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

accuracy = accuracy_score(y_test, y_pred)
print(f"Improved Neural Network Accuracy with EarlyStopping: {accuracy}")

# Сохранение модели и векторизатора
model.save('sentiment_model2.h5')
joblib.dump(vectorizer, 'vectorizer.pkl')
