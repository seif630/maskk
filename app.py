import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import os

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("D:/DEBI/mask/model.keras")

IMG_SIZE = (224, 224)

# CHANGE THIS based on your model
class_names = ["mask_weared_incorrect", "with_mask", "without_mask"]

def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)
    class_index = np.argmax(preds)
    confidence = float(np.max(preds))

    return class_names[class_index], confidence


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")

    if file is None:
        return jsonify({"error": "No image uploaded"}), 400

    file_path = "temp.jpg"
    file.save(file_path)

    label, confidence = predict_image(file_path)

    return jsonify({
        "label": label,
        "confidence": round(confidence * 100, 2)
    })


if __name__ == "__main__":
    app.run(debug=True)