from flask import Flask, request, jsonify
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load trained model and vectorizer
try:
    model = joblib.load("sentiment_model.pkl")  # Ensure this file exists
    vectorizer = joblib.load("tfidf_vectorizer.pkl")  # Ensure this file exists
    print("✅ Model and vectorizer loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model or vectorizer: {e}")
    exit(1)  # Exit if loading fails

# Sentiment mapping
sentiment_map = {
    -1: "Negative",
    0: "Neutral",
    1: "Positive"
}

# Default route to prevent 404 errors
@app.route("/")
def home():
    return jsonify({"message": "Flask Sentiment Analysis API is running!"})

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON request data
        data = request.get_json()

        # Validate input
        if "review" not in data:
            return jsonify({"error": "Missing 'review' field"}), 400

        review_text = data["review"]

        # Transform review text using TF-IDF
        review_tfidf = vectorizer.transform([review_text])

        # Make prediction
        prediction = model.predict(review_tfidf)[0]

        # Debugging: Print raw prediction output
        print(f"🔍 Raw Prediction Output: {prediction}")

        # Map prediction to sentiment label
        sentiment = sentiment_map.get(prediction, "Unknown")

        return jsonify({"review": review_text, "sentiment": sentiment})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True, port=5000)  # Change port if needed
