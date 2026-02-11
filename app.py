from flask import Flask, request, jsonify
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from utils.preprocess import preprocess_image

app = Flask(__name__)

# Load class names
with open("utils/labels.json", "r") as f:
    class_names = json.load(f)

# MODEL PATH
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "gem_identifier_model_v2.keras"

# Load model normally (no fallback needed)
model = tf.keras.models.load_model(MODEL_PATH)



@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    # Save file temporarily
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    # Preprocess image
    img = preprocess_image(file_path)


    # Predict
    predictions = model.predict(img)
    print("Prediction raw:", predictions)

    confidence = float(np.max(predictions))
    class_index = int(np.argmax(predictions))
    gem_type = class_names[class_index]

    # Delete temp file
    os.remove(file_path)

    return jsonify({
        "gem_type": gem_type,
        "confidence": confidence
    })



@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Gemora ML API Running!"})

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(host="0.0.0.0", port=5001)
