import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from imblearn.under_sampling import RandomUnderSampler

# Загрузка необходимых пакетов для NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
lemmatizer = nltk.WordNetLemmatizer()

# Функция для предобработки текста
def preprocess_text(text):
    if not isinstance(text, str):
        return None
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(text)

# Функция для кодировки меток
sentiment_mapping = {'positive': 2, 'neutral': 1, 'negative': 0}
def map_sentiment(sentiment):
    return sentiment_mapping.get(sentiment, None)

# Загрузка данных
data_urls = [
    'https://raw.githubusercontent.com/Dackohn/ExcaliburAiProject/refs/heads/main/datasets/Tweets.csv',
    'https://raw.githubusercontent.com/Dackohn/ExcaliburAiProject/refs/heads/main/datasets/sentimentdataset.csv',
    'https://raw.githubusercontent.com/Dackohn/ExcaliburAiProject/refs/heads/main/datasets/IMDB-Dataset.csv',
    'https://raw.githubusercontent.com/Dackohn/ExcaliburAiProject/refs/heads/main/datasets/train.csv',
    'https://raw.githubusercontent.com/Dackohn/ExcaliburAiProject/refs/heads/main/datasets/test.csv',
    'https://raw.githubusercontent.com/Dackohn/ExcaliburAiProject/refs/heads/main/datasets/twitter_training.csv']

dataframes = []

for url in data_urls:
    try:
        df = pd.read_csv(url, encoding='unicode_escape')
        dataframes.append(df)
    except Exception as e:
        print(f"Error loading {url}: {e}")

# Приведение к общему виду
dataframes[0] = dataframes[0][['text', 'airline_sentiment']].rename(columns={'airline_sentiment': 'sentiment'})
dataframes[1] = dataframes[1][['Text', 'Sentiment']].rename(columns={'Text': 'text', 'Sentiment': 'sentiment'})
dataframes[2] = dataframes[2].rename(columns={'review': 'text', 'sentiment': 'sentiment'})
dataframes[3] = dataframes[3][['text', 'sentiment']]
dataframes[4] = dataframes[4][['text', 'sentiment']]
dataframes[5].columns = ['Coloana1', 'Coloana2', 'sentiment', 'text']
dataframes[5] = dataframes[5][['text', 'sentiment']]

# Объединение данных
data_combined = pd.concat(dataframes, ignore_index=True)
data_combined['text'] = data_combined['text'].astype(str)
data_combined['text'] = data_combined['text'].apply(preprocess_text)
data_combined['sentiment'] = data_combined['sentiment'].map(map_sentiment)
data_combined.dropna(subset=['sentiment'], inplace=True)

# Балансировка данных
X = data_combined['text'].values
X = X.reshape(-1, 1)
y = data_combined['sentiment']
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X, y)

data_balanced = pd.DataFrame({
    'text': X_res.flatten(),
    'sentiment': y_res
})

# Word2Vec
print("Training Word2Vec model...")
tokenized_text = [text.split() for text in data_balanced['text']]
word2vec_model = Word2Vec(sentences=tokenized_text, vector_size=30, window=3, min_count=1, workers=2)

# Получение векторов
def get_average_word2vec(tokens, model, vector_size=30):
    vector = np.zeros(vector_size)
    num_tokens = 0
    for word in tokens:
        if word in model.wv:
            vector += model.wv[word]
            num_tokens += 1
    return vector / num_tokens if num_tokens > 0 else vector

X_word2vec = np.array([get_average_word2vec(text.split(), word2vec_model) for text in data_balanced['text']])

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X_word2vec, data_balanced['sentiment'], test_size=0.2, random_state=42)

# Модели
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(random_state=42),
    "Neural Network": Sequential()
}

models["Neural Network"].add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
models["Neural Network"].add(Dropout(0.2))
models["Neural Network"].add(Dense(64, activation='relu'))
models["Neural Network"].add(Dense(32, activation='relu'))
models["Neural Network"].add(Dense(3, activation='softmax'))
models["Neural Network"].compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

results = {}
for model_name, model in models.items():
    print(f"Training {model_name}...")
    if model_name == "Neural Network":
        model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test), verbose=1)
        y_pred = model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {model_name}: {acc}")
    results[model_name] = acc

# Сравнение точности
print("Model Comparison:", results)
