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
        'model_path': 'Detection_model/Tomato_LDD_model.h5',
        'class_labels': [
            'Tomato_mosaic_virus', 'Target_Spot', 'Bacterial_spot',
            'Tomato_Yellow_Leaf_Curl_Virus', 'Late_blight', 'Leaf_Mold',
            'Early_blight', 'Spider_mites Two-spotted_spider_mite',
            'Tomato___healthy', 'Septoria_leaf_spot'
        ],
        'recommendations': {
            'Tomato_mosaic_virus': {
                'text': "Use resistant plant varieties and control aphids.",
                'image': 'static/images/Tomato_mosaic_virus.jpg'
            },
            'Target_Spot': {
                'text': "Use fungicides and remove infected leaves.",
                'image': 'static/images/Target_Spot.jpg'
            },
            'Bacterial_spot': {
                'text': "Apply copper-based bactericides and manage plant spacing.",
                'image': 'static/images/Bacterial_spot.jpg'
            },
            'Tomato_Yellow_Leaf_Curl_Virus': {
                'text': "Use insecticidal soaps for whitefly control.",
                'image': 'static/images/Tomato_Yellow_Leaf_Curl_Virus.jpg'
            },
            'Late_blight': {
                'text': "Apply fungicides and practice crop rotation.",
                'image': 'static/images/Late_blight.jpg'
            },
            'Leaf_Mold': {
                'text': "Improve ventilation and apply appropriate fungicides.",
                'image': 'static/images/Leaf_Mold.jpg'
            },
            'Early_blight': {
                'text': "Use disease-resistant varieties and apply fungicides.",
                'image': 'static/images/Early_blight.jpg'
            },
            'Spider_mites Two-spotted_spider_mite': {
                'text': "Use insecticidal soap and encourage predatory mites.",
                'image': 'static/images/Spider_mites.jpg'
            },
            'Tomato___healthy': {
                'text': "No action required, plant is healthy!",
                'image': 'static/images/Tomato_healthy.jpg'
            },
            'Septoria_leaf_spot': {
                'text': "Apply fungicides and remove infected plant debris.",
                'image': 'static/images/Septoria_leaf_spot.jpg'
            }
        },
        'medicines': {
            'Tomato_mosaic_virus': 'mosaicVirus_tomato.jpeg',
            'Target_Spot': 'targetSpot_tomato.png',
            'Bacterial_spot': 'bacterialSpot_tomato.png',
            'Tomato_Yellow_Leaf_Curl_Virus': 'yellowLeafCurl_tomato.png',
            'Late_blight': 'lateBlight_tomato.jpeg',
            'Leaf_Mold': 'leafMold_tomato.jpeg',
            'Early_blight': 'earlyBlight_tomato.jpeg',
            'Spider_mites Two-spotted_spider_mite': 'spiderMites_tomato.jpeg',
            'Tomato___healthy': None,
            'Septoria_leaf_spot': 'septoriaLeafSpot_tomato.jpeg'
        }
    },
    'corn': {
        'model_path': 'Detection_model/Corn_LDD_model.h5',
        'class_labels': [
            'Cercospora leaf spot (Gray leaf spot)',
            'Common rust',
            'Northern Leaf Blight',
            'healthy'
        ],
        'recommendations': {
            'Cercospora leaf spot (Gray leaf spot)': {
                'text': "Hybrids with partial resistance to GLS are available. Ask your seed supplier for these hybrids. A two-year crop rotation away from corn is effective if reduced tillage must be maintained for conservation purposes, or a one-year rotation with clean plowing is recommended in fields that have had a problem with the disease.",
                'image': 'Cercospora_leaf_spot.jpg'
            },
            'Common rust': {
                'text': "Use resistant/tolerant sweet corn products. Many sweet corn products have resistance genes that provide nearly complete control. Applying strobilurin- and sterol-inhibiting fungicides as a preventive measure.",
                'image': 'Common_rust.jpg'
            },
            'Northern Leaf Blight': {
                'text': "Management of Northern Leaf Blight can be achieved primarily by using hybrids with resistance, but because resistance may not be complete or may fail, it is advantageous to utilize an integrated approach with different cropping practices and fungicides.",
                'image': 'Northern_Leaf_Blight.jpeg'
            },
            'healthy': {
                'text': "No action required, plant is healthy! :)",
                'image': 'healthy_corn.jpg'
            }
        },
        'medicines': {
            'Cercospora leaf spot (Gray leaf spot)': 'grayLeafSpot_corn.png',
            'Common rust': 'commonRust_corn.png',
            'Northern Leaf Blight': 'northernLeafBlight_corn.jpg',
            'healthy': None
        }
    },
    'potato': {
        'model_path': 'Detection_model/Potato_LDD_model.h5',  
        'class_labels': [
            'Potato__Early_blight', 'Potato__Late_blight', 'Potato__healthy'
        ],
        'recommendations': {
            'Potato__Early_blight': {
                'text': "Apply fungicides and remove affected leaves. Ensure proper crop rotation.",
                'image': 'static/images/Potato_Early_blight.jpg'
            },
            'Potato__Late_blight': {
                'text': "Apply fungicides immediately after symptoms appear, and remove infected plant material.",
                'image': 'static/images/Potato_Late_blight.jpg'
            },
            'Potato__healthy': {
                'text': "No action required, plant is healthy! :)",
                'image': 'static/images/Potato_healthy.jpg'
            }
        },
        'medicines': {
            'Potato__Early_blight': 'earlyBlight_potato.png',
            'Potato__Late_blight': 'lateBlight_potato.png',
            'Potato__healthy': None
        }
    },
    'grapes': {
        'model_path': 'Detection_model/Grapes_LDD_model.h5',
        'class_labels': [
            'Black Rot',
            'Esca (Black Measles)',
            'Leaf Blight (Isariopsis Leaf Spot)',
            'healthy'
        ],
        'recommendations': {
            'Black Rot': {
                'text': "Action: Remove and destroy infected leaves or fruit to reduce fungal spread. Apply fungicides such as mancozeb or myclobutanil during the growing season.",
                'image': 'static/images/Black_Rot.jpg'
            },
            'Esca (Black Measles)': {
                'text': "Prune infected wood during the dormant season. Avoid injuring the vines to prevent fungal entry. If infection is severe, consider removing the vine.",
                'image': 'static/images/Esca_Black_Measles.jpg'
            },
            'Leaf Blight (Isariopsis Leaf Spot)': {
                'text': "Improve air circulation around the grapevine by pruning overcrowded leaves. Apply a copper-based fungicide to control the spread.",
                'image': 'static/images/Leaf_Blight.jpg'
            },
            'healthy': {
                'text': "No action required, plant is healthy! :)",
                'image': 'static/images/Grapes_healthy.jpg'
            }
        },
        'medicines': {
            'Black Rot': 'blackRot_grape.jpeg',
            'Esca (Black Measles)': 'esca_grape.jpeg',
            'Leaf Blight (Isariopsis Leaf Spot)': 'leafBlight_grape.jpeg',
            'healthy': None
        }
    },
    'cotton': {
        'model_path': 'Detection_model/Cotton_LDD_model.h5',
        'class_labels': [
            'diseased cotton leaf',
            'diseased cotton plant',
            'fresh cotton leaf',
            'fresh cotton plant'
        ],
        'recommendations': {
            'diseased cotton leaf': {
                'text': "Remove infected leaves and dispose of them away from the field to prevent the spread of pathogens. Apply fungicides such as copper-based or systemic fungicides to control the disease. Ensure proper plant spacing to improve air circulation and reduce humidity.",
                'image': 'static/images/diseased_cotton_leaf.jpg'
            },
            'diseased cotton plant': {
                'text': "Isolate and destroy severely diseased plants to prevent the spread of infection. Treat the soil with biofungicides or chemical fungicides. Implement Integrated Pest Management (IPM) practices, including crop rotation and the use of resistant varieties. Maintain soil health with organic matter and balanced fertilizers.",
                'image': 'static/images/diseased_cotton_plant.jpg'
            },
            'fresh cotton leaf': {
                'text': "Maintain good cultural practices like balanced fertilization, regular irrigation, and weed control. Use preventive fungicide sprays to protect against potential diseases. Monitor leaves for pests like aphids or whiteflies, and take prompt action if detected.",
                'image': 'static/images/fresh_cotton_leaf.jpg'
            },
            'fresh cotton plant': {
                'text': "Support healthy plant growth with proper fertilization, including nitrogen, phosphorus, and potassium. Regularly monitor for pests such as bollworms and apply control measures as needed. Maintain a consistent irrigation schedule and use mulch to retain soil moisture and suppress weeds.",
                'image': 'static/images/fresh_cotton_plant.jpg'
            }
        },
        'medicines': {
            'diseased cotton leaf': 'diseasedCottonLeaf.jpeg',
            'diseased cotton plant': 'diseasedCottonLeaf.jpeg',
            'fresh cotton leaf': None,
            'fresh cotton plant': None
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
    medicines = details['medicines']

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
    recommendation = recommendations.get(predicted_class, {"text": "No recommendation available."})

    return render_template('result.html', crop=crop, disease=predicted_class, crop_data=crop_details[crop])


if __name__ == '__main__':
    app.run(debug=True)

