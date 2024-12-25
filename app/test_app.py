# test_finalmodel.py

import pytest
import json
from app import app

def test_analyze_positive_sentiment():
    client = app.test_client()
    response = client.post(
        '/analyze',
        data=json.dumps({'text': 'I love this product!'}),
        content_type='application/json'
    )
    assert response.status_code == 200
    json_data = json.loads(response.data)
    assert json_data['sentiment'] == 'positive'
