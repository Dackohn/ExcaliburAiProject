from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import joblib

# Download necessary NLTK corpora
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)
CORS(app)

# Load the Keras model
model = tf.keras.models.load_model('sentiment_model.h5')

# Load the vectorizer using joblib
vectorizer = joblib.load('vectorizer.pkl')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# Preprocessing function to clean and lemmatize the text
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    text = ' '.join([
        lemmatizer.lemmatize(word)
        for word in text.split() if word not in stop_words
    ])
    return text


@app.route('/analyze', methods=['POST'])
def analyze():
    # Get the JSON data from the request
    data = request.json
    text = data.get('text', '')

    # Check if the input text is provided
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Preprocess the text
    preprocessed_text = preprocess_text(text)

    # Transform the text into TF-IDF
    # features (ensure the correct number of features)
    try:
        X = vectorizer.transform([preprocessed_text]).toarray()
        print(f"Features extracted: {X.shape[1]}")
    except Exception as e:
        return jsonify({
            'error': f"Error during feature extraction: {str(e)}"
        }), 400

    # Check if the extracted features have the expected number (5000)
    if X.shape[1] != 5000:
        return jsonify({
            'error': f"Expected input with 5000 features, "
                     f"but got {X.shape[1]} features."
        }), 400

    # Predict sentiment with the Keras model
    try:
        prediction = model.predict(X)
        # Get the index with the highest probability
        sentiment_class = prediction.argmax(axis=-1)[0]
        sentiment_map = {
            2: 'positive',
            1: 'neutral',
            0: 'negative'
        }
        return jsonify({
            'sentiment': sentiment_map[sentiment_class],
            'score': int(sentiment_class)
        })
    except Exception as e:
        return jsonify({
            'error': f"Error during prediction: {str(e)}"
        }), 500


if __name__ == '__main__':
    app.run(debug=True)
