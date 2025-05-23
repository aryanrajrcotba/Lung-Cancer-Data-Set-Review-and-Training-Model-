from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
model = load_model('lung_cancer_model.h5')
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def predict_lung_cancer(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img) / 255.0
    img_tensor = np.expand_dims(img_tensor, axis=0)
    probability = model.predict(img_tensor)[0][0]
    status = "Not Safe" if probability > 0.5 else "Safe"
    return round(float(probability) * 100, 2), status

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            prob, status = predict_lung_cancer(filepath)
            result = f"Lung Cancer Probability: {prob}% | Status: {status}"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
