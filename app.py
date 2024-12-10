from flask import Flask, request, render_template, jsonify, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Define a dictionary of models and crop-specific details
crop_details = {
    'tomato': {
        'model_path': '/Users/kritika/Documents/MinorProjectFinal/Minor-Project/Detection_model/Tomato_LDD_model.h5',
        'class_labels': [
            'Tomato_mosaic_virus', 'Target_Spot', 'Bacterial_spot',
            'Tomato_Yellow_Leaf_Curl_Virus', 'Late_blight', 'Leaf_Mold',
            'Early_blight', 'Spider_mites Two-spotted_spider_mite',
            'Tomato___healthy', 'Septoria_leaf_spot'
        ],
        'recommendations': {
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
    },
    # Add similar dictionaries for other crops
    'corn': {
        'model_path': '/Users/kritika/Documents/MinorProjectFinal/Minor-Project/Detection_model/Maize_LDD_model.h5',
        'class_labels': ['Cercospora leaf spot (Gray leaf spot)',
                'Common rust',
                'Northern Leaf Blight',
                'healthy'],
        'recommendations': {
            'Cercospora_leaf_spot_(Gray_leaf_spot)': "Hybrids with partial resistance to GLS are available. Ask your seed supplier for these hybrids. A two-year crop rotation away from corn is effective if reduced tillage must be maintained for conservation purposes, or a one-year rotation with clean plowing is recommended in fields that have had a problem with the disease.",
    'Common_rust': "Use resistant/tolerant sweet corn products. Many sweet corn products have resistance genes that provide nearly complete control. Applying strobilurin- and sterol-inhibiting fungicides as a preventive measure.",
    'Northern_Leaf_Blight': "Management of Northern Leaf Blight can be achieved primarily by using hybrids with resistance, but because resistance may not be complete or may fail, it is advantageous to utilize an integrated approach with different cropping practices and fungicides.",
    'healthy': "No action required, plant is healthy! :)"
        }
    },
    'potato': {
    'model_path': '/Users/kritika/Documents/MinorProjectFinal/Minor-Project/Detection_model/Potato_LDD_model.h5',  
    'class_labels': [
        'Potato__Early_blight', 'Potato__Late_blight', 'Potato__healthy'
    ],
    'recommendations': {
        'Potato__Early_blight': "Apply fungicides and remove affected leaves. Ensure proper crop rotation.",
        'Potato__Late_blight': "Apply fungicides immediately after symptoms appear, and remove infected plant material.",
        'Potato__healthy': "No action required, plant is healthy! :)"
    }
}


}

@app.route('/')
def index():
    return render_template('index.html')  # Main page

@app.route('/<crop>/upload')
def upload(crop):
    if crop not in crop_details:
        return "Crop not supported.", 404
    return render_template('upload.html', crop=crop)

@app.route('/<crop>/predict', methods=['POST'])
def predict(crop):
    if crop not in crop_details:
        return "Crop not supported.", 404

    details = crop_details[crop]
    model_path = details['model_path']
    class_labels = details['class_labels']
    recommendations = details['recommendations']

    # Load the crop-specific model
    model = load_model(model_path)

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

    return render_template('result.html', crop=crop, disease=predicted_class, recommendation=recommendation)

if __name__ == '__main__':
    app.run(debug=True)

