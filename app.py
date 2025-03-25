from flask import Flask, render_template, request, jsonify, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

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
        # Base model with pre-trained weights
        base_model = DenseNet169(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        
        # Add custom layers for skin condition classification
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)  # Add dropout for better generalization
        predictions = tf.keras.layers.Dense(len(SKIN_CONDITIONS), activation='softmax')(x)
        
        model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
        
        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False
            
    return model

def preprocess_skin_image(image):
    """
    Specialized preprocessing for skin condition images.
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to standard size
    image = image.resize((224, 224))
    
    # Convert to array
    img_array = img_to_array(image)
    
    # Apply color normalization
    img_array = img_array.astype(float) / 255.0
    
    # Apply contrast enhancement
    mean = np.mean(img_array)
    std = np.std(img_array)
    img_array = (img_array - mean) / (std + 1e-7)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def analyze_skin_condition(image):
    """
    Analyze skin condition using specialized model and preprocessing.
    """
    try:
        # Ensure model is loaded
        model = load_model()
        
        # Preprocess the image with specialized techniques
        processed_image = preprocess_skin_image(image)
        
        # Get model predictions
        predictions = model.predict(processed_image)
        
        # Get top prediction and confidence
        predicted_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_index])
        
        # Map to skin conditions
        conditions_mapping = {
            0: 'حب الشباب',          # Acne
            1: 'الإكزيما',           # Eczema
            2: 'الصدفية',           # Psoriasis
            3: 'البهاق',            # Vitiligo
            4: 'حساسية الجلد',       # Skin Allergy
            5: 'العد الوردي',        # Rosacea
            6: 'سرطان الجلد',        # Skin Cancer
            7: 'الطفح الجلدي'        # Rash
        }
        
        condition = conditions_mapping.get(predicted_index, 'حساسية الجلد')
        
        # Apply confidence threshold
        if confidence < 0.5:
            return 'يرجى التحقق من جودة الصورة', 0.0
        
        return condition, confidence
        
    except Exception as e:
        print(f"Error in skin condition analysis: {str(e)}")
        return 'يرجى التحقق من جودة الصورة', 0.0

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'لم يتم تحميل أي صورة'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'لم يتم اختيار أي ملف'})
        
        # Read and process the image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Get prediction with improved model
        condition, confidence = analyze_skin_condition(image)
        
        # Get condition details from your existing dictionary
        for category, conditions in SKIN_CONDITIONS.items():
            if condition in conditions:
                condition_details = conditions[condition]
                return jsonify({
                    'condition': condition,
                    'confidence': confidence,
                    'medications': condition_details['medications'],
                    'description': condition_details['description'],
                    'category': category,
                    'recommendation': 'يرجى استشارة الطبيب للتأكيد' if confidence < 0.7 else condition_details['description']
                })
        
        return jsonify({
            'condition': condition,
            'confidence': confidence,
            'medications': 'يرجى استشارة الطبيب',
            'description': 'يرجى استشارة الطبيب للحصول على تشخيص دقيق',
            'category': 'غير محدد',
            'recommendation': 'يرجى استشارة الطبيب للتأكيد'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Use environment variable for port with a default value
    port = int(os.environ.get('PORT', 5000))
    # In production, don't use debug mode
    app.run(host='0.0.0.0', port=port) 