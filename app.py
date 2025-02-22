import os
import cv2
import numpy as np
import tensorflow.lite as tflite
from flask import Flask, render_template, request, redirect, url_for

# Inisialisasi Flask
app = Flask(__name__)

# Buat folder untuk menyimpan gambar upload jika belum ada
UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load model TFLite
interpreter = tflite.Interpreter(model_path="fruit_quality_model.tflite")
interpreter.allocate_tensors()

# Ambil indeks input & output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_image(image_path):
    """ Fungsi untuk melakukan prediksi menggunakan model TFLite """
    image = cv2.imread(image_path)  # Baca gambar dengan OpenCV
    image = cv2.resize(image, (150, 150))  # Sesuaikan ukuran dengan input model
    image = np.expand_dims(image, axis=0)  # Tambahkan batch dimension
    image = image.astype(np.float32) / 255.0  # Normalisasi

    # Masukkan gambar ke model
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    
    # Ambil hasil prediksi
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
    
    return "Segar" if prediction > 0.5 else "Busuk"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(image_path)  # Simpan gambar

            # Prediksi gambar
            result = predict_image(image_path)
            
            return render_template("index.html", image_path=image_path, result=result)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
