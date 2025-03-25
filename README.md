# Skin Condition Recognition App

This application helps identify skin conditions from uploaded images and suggests appropriate medications.

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5000`

## Features

- Upload skin condition images
- AI-powered skin condition recognition
- Suggested medications based on identified conditions
- User-friendly interface

## Technical Details

- Backend: Flask (Python)
- Frontend: HTML, CSS, JavaScript
- AI Model: TensorFlow/Keras
- Image Processing: Pillow

## Note

The accuracy of predictions depends on the quality of the input images and the training data used for the model. 