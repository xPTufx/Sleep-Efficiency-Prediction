from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Load the model
model = joblib.load("sleep_efficiency_model_gradboost.joblib")

EXPECTED_FEATURES = [
    "Gender", "Age", "Sleep Duration",
    "Stress Level", "BMI Category", "Heart Rate",
    "Daily Steps", "Sleep Disorder"
]

@app.route("/", methods=["GET"])
def home():
    return "Model API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data.get(EXPECTED_FEATURES)  # Should be a list or 2D list

    try:
        prediction = model.predict([features])  # or just features if already 2D
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400