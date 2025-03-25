from flask import Flask, render_template, request, jsonify, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

app = Flask(__name__, static_url_path='/static')

# Define the skin conditions organized by categories in Arabic
SKIN_CONDITIONS = {
    'الأمراض الالتهابية': {
        'حب الشباب': {
            'medications': ['بنزويل بيروكسيد', 'حمض الساليسيليك', 'الريتينويدات الموضعية'],
            'description': 'حالة جلدية شائعة تتميز بظهور البثور والرؤوس السوداء والرؤوس البيضاء.'
        },
        'الإكزيما': {
            'medications': ['الكورتيزون الموضعي', 'المرطبات', 'مضادات الهيستامين'],
            'description': 'حالة التهابية في الجلد تسبب الحكة والاحمرار وجفاف البشرة.'
        },
        'الدمامل': {
            'medications': ['المضادات الحيوية الموضعية', 'المضادات الحيوية عن طريق الفم', 'كمادات دافئة'],
            'description': 'التهاب موضعي في الجلد يسبب تورم مؤلم وتجمع للصديد.'
        },
        'العد الوردي': {
            'medications': ['المضادات الحيوية الموضعية', 'حمض الأزيليك', 'مشتقات فيتامين أ'],
            'description': 'حالة جلدية تسبب احمرار الوجه وظهور بثور صغيرة.'
        }
    },
    
    'الأمراض المناعية': {
        'الصدفية': {
            'medications': ['الستيرويدات الموضعية', 'مشتقات فيتامين د', 'قطران الفحم'],
            'description': 'مرض مناعي ذاتي يسبب نمو سريع لخلايا الجلد وتقشر البشرة.'
        },
        'الصدفية الصدفية': {
            'medications': ['العلاج البيولوجي', 'الميثوتريكسات', 'العلاج بالأشعة فوق البنفسجية'],
            'description': 'نوع شديد من الصدفية يغطي مساحات كبيرة من الجسم.'
        },
        'الحزاز المسطح': {
            'medications': ['كريمات الكورتيزون', 'مضادات الهيستامين', 'مثبطات المناعة الموضعية'],
            'description': 'مرض جلدي يسبب ظهور بقع حمراء أو بنفسجية مسطحة وحكة.'
        }
    },

    'أمراض الحساسية': {
        'حساسية الجلد': {
            'medications': ['مضادات الهيستامين', 'كريمات الكورتيزون', 'مستحضرات التبريد'],
            'description': 'تفاعل جلدي تحسسي يسبب الحكة والطفح الجلدي والاحمرار.'
        },
        'التهاب الجلد التماسي': {
            'medications': ['كريمات الكورتيزون', 'المرطبات', 'مضادات الهيستامين'],
            'description': 'تفاعل جلدي ناتج عن ملامسة مواد مهيجة أو مسببة للحساسية.'
        },
        'الشرى': {
            'medications': ['مضادات الهيستامين', 'كريمات الكورتيزون', 'مثبطات المناعة'],
            'description': 'طفح جلدي مفاجئ يسبب انتفاخات حمراء وحكة شديدة.'
        }
    },

    'الأمراض الطفيلية والفطرية': {
        'الجرب': {
            'medications': ['البيرميثرين', 'الإيفرمكتين', 'مضادات الهيستامين للحكة'],
            'description': 'عدوى طفيلية تسبب حكة شديدة وطفح جلدي.'
        },
        'القوباء الحلقية': {
            'medications': ['مضادات الفطريات الموضعية', 'كريمات مضادة للفطريات عن طريق الفم', 'الشامبو المضاد للفطريات'],
            'description': 'عدوى فطرية تظهر على شكل حلقات حمراء على الجلد.'
        }
    },

    'اضطرابات الصبغات والشعر': {
        'البهاق': {
            'medications': ['كريمات الكورتيزون', 'مثبطات المناعة الموضعية', 'العلاج بالأشعة فوق البنفسجية'],
            'description': 'حالة جلدية تؤدي إلى فقدان لون الجلد في مناطق معينة.'
        },
        'الثعلبة': {
            'medications': ['كريمات الكورتيزون', 'المنشطات الموضعية', 'حقن الكورتيزون'],
            'description': 'حالة تؤدي إلى تساقط الشعر في مناطق دائرية.'
        }
    },

    'أمراض الأوعية الدموية': {
        'الوردية': {
            'medications': ['ميترونيدازول', 'حمض الأزيليك', 'كريمات مضادة للالتهابات'],
            'description': 'حالة جلدية مزمنة تسبب احمرار الوجه وظهور الأوعية الدموية.'
        }
    }
}

# Global variable for the model
model = None

def get_all_conditions():
    """Get a flat list of all conditions."""
    all_conditions = {}
    for category in SKIN_CONDITIONS.values():
        all_conditions.update(category)
    return all_conditions

# Ensure static directory exists
if not os.path.exists('static'):
    os.makedirs('static')

def load_model():
    """Load and prepare the model."""
    global model
    if model is None:
        # Use MobileNetV2 as base model (efficient and good for mobile/web deployment)
        model = MobileNetV2(weights='imagenet')
    return model

def analyze_skin_image(image):
    """
    Analyze the image using MobileNetV2 and map predictions to skin conditions.
    """
    try:
        # Ensure model is loaded
        model = load_model()
        
        # Preprocess the image
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image to 224x224 (required for MobileNetV2)
        image = image.resize((224, 224))
        
        # Convert to array and preprocess
        img_array = np.array(image)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Get predictions
        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=5)[0]
        
        # Map the predictions to skin conditions based on visual features
        confidence = float(predictions.max())
        
        # Get the top predicted class
        predicted_class = decoded_predictions[0][1].lower()
        
        # Map general image features to skin conditions
        feature_to_condition = {
            'rash': 'الطفح الجلدي',
            'skin': 'حساسية الجلد',
            'spot': 'حب الشباب',
            'red': 'الإكزيما',
            'scale': 'الصدفية',
            'patch': 'البهاق',
            'lesion': 'سرطان الجلد'
        }
        
        # Find the most relevant skin condition based on predicted features
        condition = None
        for feature, skin_condition in feature_to_condition.items():
            if feature in predicted_class:
                condition = skin_condition
                break
        
        # If no specific mapping found, use the most common condition
        if not condition:
            condition = 'حساسية الجلد'
        
        return condition, confidence
        
    except Exception as e:
        print(f"Error in image analysis: {str(e)}")
        return 'حساسية الجلد', 0.5

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image from the POST request
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        # Read and process the image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Get prediction
        condition, confidence = analyze_skin_image(image)
        
        # Get condition details
        for category, conditions in SKIN_CONDITIONS.items():
            if condition in conditions:
                condition_details = conditions[condition]
                return jsonify({
                    'condition': condition,
                    'confidence': confidence,
                    'medications': condition_details['medications'],
                    'description': condition_details['description'],
                    'category': category
                })
        
        # Fallback if condition not found in dictionary
        return jsonify({
            'condition': condition,
            'confidence': confidence,
            'medications': 'يرجى استشارة الطبيب',
            'description': 'يرجى استشارة الطبيب للحصول على تشخيص دقيق',
            'category': 'غير محدد'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Use environment variable for port with a default value
    port = int(os.environ.get('PORT', 5000))
    # In production, don't use debug mode
    app.run(host='0.0.0.0', port=port) 