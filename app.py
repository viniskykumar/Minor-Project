from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load the pre-trained plant disease model
model = load_model('Tomato_LDD_model (1).h5')

# Define class labels and recommendations
class_labels = [
    'Tomato_mosaic_virus', 'Target_Spot', 'Bacterial_spot',
    'Tomato_Yellow_Leaf_Curl_Virus', 'Late_blight', 'Leaf_Mold',
    'Early_blight', 'Spider_mites Two-spotted_spider_mite',
    'Tomato___healthy', 'Septoria_leaf_spot'
]

recommendations = {
    'Tomato_mosaic_virus': "Use resistant plant varieties and control aphids.",
    'Target_Spot': "Use fungicides and remove infected leaves.",
    'Bacterial_spot': "Apply copper-based bactericides and manage plant spacing.",
    'Tomato_Yellow_Leaf_Curl_Virus': "Use insecticidal soaps for whitefly control.",
    'Late_blight': "Apply fungicides and practice crop rotation.",
    'Leaf_Mold': "Improve ventilation and apply appropriate fungicides.",
    'Early_blight': "Use disease-resistant varieties and apply fungicides.",
    'Spider_mites Two-spotted_spider_mite': "Use insecticidal soap and encourage predatory mites.",
    'Tomato___healthy': "No action required, plant is healthy!",
    'Septoria_leaf_spot': "Apply fungicides and remove infected plant debris."
}

@app.route('/')
def index():
    return render_template('index.html')  # Main page

@app.route('/upload')
def upload():
    return render_template('upload.html')  # Page with image upload form

# Route for predicting plant disease
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Process the image
    image = Image.open(file).convert('RGB')
    image = image.resize((256, 256))  # Resize to model's input size
    image = img_to_array(image) / 255.0  # Normalize
    input_data = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(input_data)
    predicted_class = class_labels[np.argmax(predictions)]
    recommendation = recommendations.get(predicted_class, "No recommendation available.")

    return render_template('result.html', disease=predicted_class, recommendation=recommendation)

if __name__ == '__main__':
    app.run(debug=True)
