import os
from flask import Flask, request, jsonify
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load trained model and vectorizer
try:
    model = joblib.load("sentiment_model.pkl")  
    vectorizer = joblib.load("tfidf_vectorizer.pkl")  
    print("✅ Model and vectorizer loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model or vectorizer: {e}")
    exit(1)

# Sentiment mapping
sentiment_map = {
    -1: "Negative",
    0: "Neutral",
    1: "Positive"
}

# Default route to prevent 404 errors
@app.route("/")
def home():
    return jsonify({"message": "Flask Sentiment Analysis API is running on Render!"})

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if "review" not in data:
            return jsonify({"error": "Missing 'review' field"}), 400

        review_text = data["review"]
        review_tfidf = vectorizer.transform([review_text])
        prediction = model.predict(review_tfidf)[0]

        sentiment = sentiment_map.get(prediction, "Unknown")
        return jsonify({"review": review_text, "sentiment": sentiment})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask app on Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render's assigned port
    app.run(host="0.0.0.0", port=port, debug=True)
