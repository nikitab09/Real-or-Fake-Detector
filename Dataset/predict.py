import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("deepfake_model.h5")

img = image.load_img("test.jpg", target_size=(224,224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
confidence = prediction[0][0]

if confidence > 0.5:
    print(f"Real Image (Confidence: {confidence:.2f})")
else:
    print(f"Fake Image (Confidence: {1-confidence:.2f})")