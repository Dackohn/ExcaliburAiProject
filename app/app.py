from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

app = Flask(__name__)
CORS(app)

# Load model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    preprocessed_text = preprocess_text(text)
    X = vectorizer.transform([preprocessed_text])
    prediction = model.predict(X)[0]
    sentiment_map = {2: 'positive', 1: 'neutral', 0: 'negative'}

    return jsonify({'sentiment': sentiment_map[prediction], 'score': int(prediction)})


if __name__ == '__main__':
    app.run(debug=True)
