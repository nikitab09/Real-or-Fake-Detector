from flask import Flask, render_template, request, redirect, url_for, session
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)
app.secret_key = "deepfake_secret_key"

model = tf.keras.models.load_model("deepfake_model.h5")

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]

        if file:
            filename = file.filename
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            image_path = f"uploads/{filename}"

            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            confidence = float(prediction[0][0])

            if confidence > 0.5:
                prediction_text = f"Real Image (Confidence: {confidence:.2f})"
            else:
                prediction_text = f"Fake Image (Confidence: {1-confidence:.2f})"

            session['prediction'] = prediction_text
            session['confidence'] = confidence
            session['image_path'] = image_path

            return redirect(url_for('index'))

    prediction_text = session.get('prediction', '')
    confidence = session.get('confidence', 0)
    image_path = session.get('image_path', '')

    return render_template(
        'index.html',
        prediction=prediction_text,
        confidence=confidence,
        image_path=image_path
    )


@app.route("/reset")
def reset():
    session.clear()
    return redirect(url_for('index'))


if __name__ == "__main__":
    app.run(debug=True)