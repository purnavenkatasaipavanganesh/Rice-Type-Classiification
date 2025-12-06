from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load model
model = load_model("models/mobilenet_model.h5", compile=False)


# Class names (EDIT THESE WITH YOUR LABELS)
class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine' , 'karacadag']

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))    # change size if your model needs different
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", prediction="No file selected")

        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html", prediction="No file selected")

        # Save file temporarily
        img_path = os.path.join("static", file.filename)
        file.save(img_path)

        # Preprocess + Predict
        processed_img = preprocess_image(img_path)
        preds = model.predict(processed_img)
        class_id = np.argmax(preds)
        prediction_text = class_names[class_id]

        return render_template("index.html", prediction=prediction_text, img_path=img_path)

    return render_template("index.html", prediction=None)
    
if __name__ == "__main__":
    app.run(debug=True)
