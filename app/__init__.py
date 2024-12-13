import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load the dataset
file_path = 'D:/Users/ASUS-X509J/Desktop/ExcaliburAiProject/Tweets.csv'
df = pd.read_csv(file_path, sep=',')

# Explore the dataset
print("Dataset shape:", df.shape)
print("Columns:\n", df.columns)
print("\nSample rows:\n", df[['airline_sentiment', 'text']].head())

# Select relevant columns
columns_to_keep = ['airline_sentiment', 'text']
df = df[columns_to_keep]

# Check sentiment distribution
print("\nSentiment distribution:\n", df['airline_sentiment'].value_counts())

# Drop rows with missing values in 'text'
df = df.dropna(subset=['text'])
print("\nShape after removing rows with missing text:", df.shape)

# Preprocess the text
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text)  # Tokenize
    tokens = [word.lower() for word in tokens if word.isalnum()]  # Remove punctuation and lowercase
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize
    return " ".join(tokens)

# Apply preprocessing to the 'text' column
df['processed_text'] = df['text'].apply(preprocess_text)
print("\nSample of processed text:\n", df[['text', 'processed_text']].head())

# Check the processed sentiment and text
print("\nProcessed Data Sample:\n", df[['airline_sentiment', 'processed_text']].head())

# Save the processed dataset
output_path = 'D:/Users/ASUS-X509J/Desktop/ExcaliburAiProject/Tweets_processed.csv'
df.to_csv(output_path, index=False)
print(f"\nProcessed dataset saved successfully to {output_path}.")
