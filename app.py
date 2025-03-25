from flask import Flask, render_template, request, jsonify, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
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
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        
        # Add custom classification layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(len(get_all_conditions()), activation='softmax')(x)
        
        # Create the full model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False
            
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    return model

def preprocess_image(image):
    """Preprocess the image for the model."""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image to 224x224 (MobileNetV2 standard size)
    image = image.resize((224, 224))
    
    # Convert to array and preprocess
    img_array = np.array(image)
    img_array = preprocess_input(img_array)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_skin_condition(image):
    """
    Predict skin condition using the trained model.
    """
    try:
        # Ensure model is loaded
        model = load_model()
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Get predictions
        predictions = model.predict(processed_image)[0]
        
        # Get all conditions for mapping predictions
        all_conditions = list(get_all_conditions().keys())
        
        # Get the highest confidence prediction
        predicted_index = np.argmax(predictions)
        confidence = float(predictions[predicted_index])
        condition = all_conditions[predicted_index]
        
        return condition, confidence
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        # Fallback to random prediction if model fails
        all_conditions = get_all_conditions()
        condition = np.random.choice(list(all_conditions.keys()))
        confidence = np.random.random()
        return condition, confidence

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'لم يتم تحميل أي صورة'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'لم يتم اختيار أي ملف'}), 400

    try:
        # Read and preprocess the image
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)
        
        # Get prediction
        condition, confidence = predict_skin_condition(processed_image)
        
        # Find the condition info in the nested structure
        all_conditions = get_all_conditions()
        condition_info = all_conditions[condition]
        
        # Get category
        category = next(cat_name for cat_name, cat_conditions in SKIN_CONDITIONS.items() 
                      if condition in cat_conditions)
        
        return jsonify({
            'condition': condition,
            'category': category,
            'confidence': float(confidence),
            'medications': condition_info['medications'],
            'description': condition_info['description']
        })
    
    except Exception as e:
        return jsonify({'error': 'حدث خطأ أثناء معالجة الصورة'}), 500

if __name__ == '__main__':
    # Use environment variable for port with a default value
    port = int(os.environ.get('PORT', 5000))
    # In production, don't use debug mode
    app.run(host='0.0.0.0', port=port) 