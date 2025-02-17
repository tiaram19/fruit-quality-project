from flask import Flask, request, render_template
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = load_model("fruit_quality_model.h5")  # Pastikan model sudah dilatih dan disimpan

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# def predict_fruit(img_path):
#     img = image.load_img(img_path, target_size=(150, 150))  # Sesuaikan dengan ukuran input model
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalisasi
#     prediction = model.predict(img_array)
#     return "Segar" if prediction[0][0] > 0.5 else "Busuk"
def predict_fruit(img_path):
    img = image.load_img(img_path, target_size=(150, 150))  # Sesuaikan dengan input model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalisasi
    prediction = model.predict(img_array)
    
    prob = prediction[0][0]  # Ambil nilai probabilitas
    print(f"Probabilitas: {prob}")  # Debugging
    
    return "Segar" if prob < 0.5 else "Busuk"  # Tukar tanda `<` dan `>`


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            result = predict_fruit(filepath)
            return render_template('index.html', result=result, image_path=filepath)
    return render_template('index.html', result=None, image_path=None)

if __name__ == '__main__':
    app.run(debug=True)
