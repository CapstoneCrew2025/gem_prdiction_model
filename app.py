import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from werkzeug.utils import secure_filename

# -----------------------------
# Flask App Setup
# -----------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8 MB upload limit

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = "gem_final_model (1).keras"  # replace with your actual model file
model = load_model(MODEL_PATH)

# -----------------------------
# Class Names (Hardcoded)
# -----------------------------
CLASS_NAMES = [
    'Alexandrite', 'Almandine', 'Amazonite', 'Amber', 'Amethyst', 'Ametrine', 'Andalusite', 'Andradite',
    'Aquamarine', 'Aventurine Green', 'Aventurine Yellow', 'Benitoite', 'Beryl Golden', 'Bixbite',
    'Bloodstone', 'Blue Lace Agate', 'Carnelian', 'Cats Eye', 'Chalcedony', 'Chalcedony Blue',
    'Chrome Diopside', 'Chrysoberyl', 'Chrysocolla', 'Chrysoprase', 'Citrine', 'Coral', 'Danburite',
    'Diamond', 'Diaspore', 'Dumortierite', 'Emerald', 'Fluorite', 'Garnet Red', 'Goshenite', 'Grossular',
    'Hessonite', 'Hiddenite', 'Iolite', 'Jade', 'Jasper', 'Kunzite', 'Kyanite', 'Labradorite',
    'Lapis Lazuli', 'Larimar', 'Malachite', 'Moonstone', 'Morganite', 'Onyx Black', 'Onyx Green',
    'Onyx Red', 'Opal', 'Pearl', 'Peridot', 'Prehnite', 'Pyrite', 'Pyrope', 'Quartz Beer', 'Quartz Lemon',
    'Quartz Rose', 'Quartz Rutilated', 'Quartz Smoky', 'Rhodochrosite', 'Rhodolite', 'Rhodonite', 'Ruby',
    'Sapphire Blue', 'Sapphire Pink', 'Sapphire Purple', 'Sapphire Yellow', 'Scapolite', 'Serpentine',
    'Sodalite', 'Spessartite', 'Sphene', 'Spinel', 'Spodumene', 'Sunstone', 'Tanzanite', 'Tigers Eye',
    'Topaz', 'Tourmaline', 'Tsavorite', 'Turquoise', 'Variscite', 'Zircon', 'Zoisite'
]

IMG_SIZE = (300, 300)

# -----------------------------
# Home Page (Upload Form)
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("upload.html")  # Upload form HTML

# -----------------------------
# Prediction Route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    # Save uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Load and preprocess
    img = load_img(file_path, target_size=IMG_SIZE)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Predict
    preds = model.predict(x)[0]
    top_idx = np.argmax(preds)
    confidence = float(preds[top_idx]) * 100

    return render_template("result.html",
                           image_path=file_path,
                           label=CLASS_NAMES[top_idx],
                           confidence=round(confidence, 2))

# -----------------------------
# Run the App
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
