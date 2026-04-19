from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("model.h5")

# Labels (36 classes)
labels = ['0','1','2','3','4','5','6','7','8','9',
          'a','b','c','d','e','f','g','h','i','j',
          'k','l','m','n','o','p','q','r','s','t',
          'u','v','w','x','y','z']

# Prepare uploaded image
def prepare_image(img):
    img = img.convert("RGB")
    img = img.resize((64, 64))
    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Home page
@app.route('/')
def home():
    return render_template("index.html")

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = Image.open(file)

    processed = prepare_image(img)

    prediction = model.predict(processed)[0]

    # Top 3 predictions
    top3 = prediction.argsort()[-3:][::-1]

    result = ", ".join(
        [f"{labels[i]} ({prediction[i]*100:.1f}%)" for i in top3]
    )

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)