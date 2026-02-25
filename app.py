from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("deepfake_model.h5")

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create uploads folder if not exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = ""
    image_path = ""

    if request.method == "POST":
        file = request.files["file"]

        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            image_path = filepath

            # Preprocess image
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Prediction
            prediction = model.predict(img_array)
            confidence = prediction[0][0]

            if confidence > 0.5:
                prediction_text = f"Real Image (Confidence: {confidence:.2f})"
            else:
                prediction_text = f"Fake Image (Confidence: {1-confidence:.2f})"

    return render_template("index.html",
                           prediction=prediction_text,
                           image_path=image_path)


if __name__ == "__main__":
    app.run(debug=True)