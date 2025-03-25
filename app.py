from flask import Flask, render_template, request, jsonify, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from tensorflow.keras.applications import DenseNet169, ResNet50V2, EfficientNetB4
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.preprocessing.image import img_to_array
from skimage import exposure, segmentation

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

# Initialize models globally
models = {
    'densenet': None,
    'resnet': None,
    'efficientnet': None
}

def get_all_conditions():
    """Get a flat list of all conditions."""
    all_conditions = {}
    for category in SKIN_CONDITIONS.values():
        all_conditions.update(category)
    return all_conditions

# Ensure static directory exists
if not os.path.exists('static'):
    os.makedirs('static')

def load_models():
    """Load multiple models for ensemble prediction."""
    global models
    
    if models['densenet'] is None:
        # DenseNet for detailed feature extraction
        base_model = DenseNet169(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        predictions = tf.keras.layers.Dense(len(SKIN_CONDITIONS), activation='softmax')(x)
        models['densenet'] = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    
    if models['resnet'] is None:
        # ResNet for hierarchical feature learning
        base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        predictions = tf.keras.layers.Dense(len(SKIN_CONDITIONS), activation='softmax')(x)
        models['resnet'] = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    
    if models['efficientnet'] is None:
        # EfficientNet for mobile-optimized inference
        base_model = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        predictions = tf.keras.layers.Dense(len(SKIN_CONDITIONS), activation='softmax')(x)
        models['efficientnet'] = tf.keras.Model(inputs=base_model.input, outputs=predictions)

def segment_skin_region(image):
    """Segment the skin region from the image."""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to array
    img_array = np.array(image)
    
    # Apply SLIC segmentation
    segments = segmentation.slic(img_array, n_segments=100, compactness=10)
    
    # Create mask for skin-colored regions
    skin_mask = np.zeros(img_array.shape[:2], dtype=bool)
    
    for segment_id in np.unique(segments):
        segment = segments == segment_id
        segment_color = img_array[segment].mean(axis=0)
        
        # Skin color detection in RGB space
        r, g, b = segment_color
        is_skin = (r > 95 and g > 40 and b > 20 and
                  max(r, g, b) - min(r, g, b) > 15 and
                  abs(r - g) > 15 and r > g and r > b)
        
        if is_skin:
            skin_mask[segment] = True
    
    # Apply mask to original image
    segmented_image = img_array.copy()
    segmented_image[~skin_mask] = [0, 0, 0]
    
    return Image.fromarray(segmented_image)

def enhance_image_quality(image):
    """Enhance image quality for better feature detection."""
    # Convert to array
    img_array = np.array(image)
    
    # Enhance contrast using adaptive histogram equalization
    img_array_lab = exposure.equalize_adapthist(img_array)
    
    # Denoise
    img_array_denoised = exposure.denoise_bilateral(img_array_lab)
    
    return Image.fromarray((img_array_denoised * 255).astype(np.uint8))

def get_ensemble_prediction(image):
    """Get predictions from multiple models and combine them."""
    predictions = []
    
    # Preprocess image for each model
    img_densenet = densenet_preprocess(np.array(image.resize((224, 224))))
    img_resnet = resnet_preprocess(np.array(image.resize((224, 224))))
    img_efficientnet = efficientnet_preprocess(np.array(image.resize((224, 224))))
    
    # Get predictions from each model
    pred_densenet = models['densenet'].predict(np.expand_dims(img_densenet, axis=0))
    pred_resnet = models['resnet'].predict(np.expand_dims(img_resnet, axis=0))
    pred_efficientnet = models['efficientnet'].predict(np.expand_dims(img_efficientnet, axis=0))
    
    # Weighted average of predictions (can be adjusted based on model performance)
    weights = [0.4, 0.3, 0.3]  # DenseNet, ResNet, EfficientNet
    ensemble_pred = (weights[0] * pred_densenet +
                    weights[1] * pred_resnet +
                    weights[2] * pred_efficientnet)
    
    return ensemble_pred[0]

def analyze_skin_condition(image):
    """Enhanced skin condition analysis with multiple models and preprocessing."""
    try:
        # Load models if not loaded
        load_models()
        
        # Segment skin region
        segmented_image = segment_skin_region(image)
        
        # Enhance image quality
        enhanced_image = enhance_image_quality(segmented_image)
        
        # Get ensemble predictions
        predictions = get_ensemble_prediction(enhanced_image)
        
        # Get top prediction and confidence
        predicted_index = np.argmax(predictions)
        confidence = float(predictions[predicted_index])
        
        # Enhanced confidence calculation
        confidence_threshold = 0.6
        secondary_confidence = sorted(predictions)[-2]  # Second highest confidence
        confidence_margin = confidence - secondary_confidence
        
        # Adjust confidence based on prediction margin
        adjusted_confidence = confidence * (1 + confidence_margin)
        
        # Map to skin conditions with enhanced logic
        conditions_mapping = {
            0: {'name': 'حب الشباب', 'severity': ['خفيف', 'متوسط', 'شديد']},
            1: {'name': 'الإكزيما', 'severity': ['خفيف', 'متوسط', 'شديد']},
            2: {'name': 'الصدفية', 'severity': ['خفيف', 'متوسط', 'شديد']},
            3: {'name': 'البهاق', 'severity': ['محدود', 'متوسط', 'منتشر']},
            4: {'name': 'حساسية الجلد', 'severity': ['خفيف', 'متوسط', 'شديد']},
            5: {'name': 'العد الوردي', 'severity': ['مبكر', 'متوسط', 'متقدم']},
            6: {'name': 'سرطان الجلد', 'severity': ['مبكر', 'متوسط', 'متقدم']},
            7: {'name': 'الطفح الجلدي', 'severity': ['خفيف', 'متوسط', 'شديد']}
        }
        
        condition_info = conditions_mapping.get(predicted_index)
        
        # Determine severity based on confidence
        severity_index = 0
        if adjusted_confidence > 0.8:
            severity_index = 2
        elif adjusted_confidence > 0.6:
            severity_index = 1
        
        condition = condition_info['name']
        severity = condition_info['severity'][severity_index]
        
        # Return enhanced results
        return {
            'condition': condition,
            'confidence': adjusted_confidence,
            'severity': severity,
            'needs_urgent_care': severity_index == 2 or condition == 'سرطان الجلد'
        }
        
    except Exception as e:
        print(f"Error in skin condition analysis: {str(e)}")
        return {
            'condition': 'يرجى التحقق من جودة الصورة',
            'confidence': 0.0,
            'severity': 'غير محدد',
            'needs_urgent_care': False
        }

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
        
        # Get enhanced prediction
        result = analyze_skin_condition(image)
        
        # Get condition details from your existing dictionary
        for category, conditions in SKIN_CONDITIONS.items():
            if result['condition'] in conditions:
                condition_details = conditions[result['condition']]
                return jsonify({
                    'condition': result['condition'],
                    'confidence': result['confidence'],
                    'severity': result['severity'],
                    'needs_urgent_care': result['needs_urgent_care'],
                    'medications': condition_details['medications'],
                    'description': condition_details['description'],
                    'category': category,
                    'recommendation': 'يرجى استشارة الطبيب فوراً' if result['needs_urgent_care'] 
                                    else 'يرجى استشارة الطبيب للتأكيد' if result['confidence'] < 0.7 
                                    else condition_details['description']
                })
        
        return jsonify({
            'condition': result['condition'],
            'confidence': result['confidence'],
            'severity': result['severity'],
            'needs_urgent_care': result['needs_urgent_care'],
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