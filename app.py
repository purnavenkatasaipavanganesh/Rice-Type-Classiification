from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import os

app = Flask(__name__)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="models/mobilenet_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'karacadag']

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array.astype(np.float32)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None
    img_path = None

    if request.method == "POST":
        if "file" not in request.files or request.files["file"].filename == "":
            return render_template("index.html", prediction="No file selected")

        file = request.files["file"]

        img_path = os.path.join("static", file.filename)
        file.save(img_path)

        input_data = preprocess_image(img_path)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        class_id = np.argmax(output_data)
        prediction_text = class_names[class_id]

        return render_template("index.html", prediction=prediction_text, img_path=img_path)

    return render_template("index.html", prediction=None)
    
if __name__ == "__main__":
    app.run(debug=True)
