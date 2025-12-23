from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import re

app = Flask(__name__)
CORS(app)

# Load trained model
with open('api/url_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Feature Extraction Logic
def extract_features(url):
    return [
        len(url),
        url.count('.'),
        url.count('-'),
        url.count('@'),
        url.count('?'),
        url.count('='),
        url.count('/'),
        sum(c.isdigit() for c in url),
        int(bool(re.search(r'\b\d{1,3}(\.\d{1,3}){3}\b', url))), 
        int("https" in url.lower()),
        int("login" in url.lower()),
        int("secure" in url.lower()),
        int("account" in url.lower())
    ]

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        url = data.get('url', '')
        
        if not url:
            return jsonify({"error": "No URL provided"}), 400

        # Convert URL to features
        features = np.array([extract_features(url)], dtype=np.float32)
        
        # Get Prediction
        prediction = model.predict(features)[0]
        
        # Get Confidence (LogisticRegression)
        try:
            proba = model.predict_proba(features)[0]
            confidence = float(np.max(proba) * 100)
        except:
            confidence = 100.0 

        return jsonify({
            "isPhishing": bool(prediction == 1), # Assumes 1 is phishing, 0 is safe
            "confidence": confidence,
            "threatType": "Phishing" if prediction == 1 else "Safe",
            "analysisTime": 0.5, 
            "recommendations": [
                "Do not enter personal info", "Report this URL"
            ] if prediction == 1 else ["URL appears safe", "Stay vigilant"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)