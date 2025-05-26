from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
model = load_model('lung_cancer_model.h5')
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Class labels
CLASS_LABELS = ['Adenocarcinoma', 'Normal', 'Squamous Cell Carcinoma']

def predict_lung_cancer(img_path):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img) / 255.0
    img_tensor = np.expand_dims(img_tensor, axis=0)
    
    # Get predictions
    predictions = model.predict(img_tensor)[0]
    
    # Get the predicted class and its probability
    predicted_class = np.argmax(predictions)
    probability = predictions[predicted_class]
    
    # Get all probabilities
    probabilities = {
        label: round(float(prob) * 100, 2)
        for label, prob in zip(CLASS_LABELS, predictions)
    }
    
    # Determine status
    if predicted_class == 1:  # Normal
        status = "Normal"
    else:  # Adenocarcinoma or Squamous Cell Carcinoma
        status = "Abnormal"
    
    return probabilities, status, CLASS_LABELS[predicted_class]

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    probabilities = None
    status = None
    predicted_class = None
    
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Save the uploaded file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            
            probabilities, status, predicted_class = predict_lung_cancer(filepath)
            
            
            os.remove(filepath)
    
    return render_template('index.html', 
                         result=result,
                         probabilities=probabilities,
                         status=status,
                         predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
