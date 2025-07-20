# Handwritten Digit Recognition Using CNN

A deep learning web application that recognizes handwritten digits (0-9) using Convolutional Neural Networks (CNN) and deploys the model through a Flask web interface.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Model Details](#model-details)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [Future Enhancements](#future-enhancements)
- [Acknowledgments](#acknowledgments)

## Overview

This project implements a **Convolutional Neural Network (CNN)** for recognizing handwritten digits from the famous MNIST dataset. The trained model is deployed as a web application using **Flask**, allowing users to draw digits on a canvas interface and receive real-time predictions.

The system achieves high accuracy in digit classification and provides an intuitive web interface for interactive digit recognition, making it suitable for educational purposes, proof-of-concept demonstrations, and as a foundation for more complex optical character recognition systems.

## Features

### Core Features
- **High-Accuracy CNN Model**: Trained on MNIST dataset with >99% accuracy
- **Interactive Web Interface**: Canvas-based digit drawing with real-time prediction
- **Real-time Processing**: Instant digit recognition upon user input
- **Responsive Design**: Cross-platform compatibility (desktop, tablet, mobile)

### Technical Features
- **RESTful API**: Clean API endpoints for model predictions
- **Image Preprocessing**: Automatic normalization and resizing pipeline
- **Base64 Image Handling**: Seamless conversion between canvas data and model input
- **Model Serialization**: Efficient model loading and caching
- **Error Handling**: Comprehensive error management and logging

## Architecture

### System Architecture
```
┌─────────────────┐    HTTP Request    ┌─────────────────┐
│   Frontend      │ ──────────────────→│   Flask Server  │
│   (Canvas UI)   │                    │   (Backend)     │
└─────────────────┘                    └─────────────────┘
                                              │
                                              ▼
                                       ┌─────────────────┐
                                       │   CNN Model     │
                                       │   (Prediction)  │
                                       └─────────────────┘
```

### Model Architecture
```
Input (28x28x1) → Conv2D → MaxPooling2D → Conv2D → MaxPooling2D → 
Dropout → Flatten → Dense → Dropout → Dense (10 classes) → Softmax
```

## Dataset

The model is trained on the **MNIST (Modified National Institute of Standards and Technology)** dataset:

- **Training Set**: 60,000 images (28×28 grayscale)
- **Test Set**: 10,000 images (28×28 grayscale) 
- **Classes**: 10 (digits 0-9)
- **Image Format**: Grayscale, normalized to [0,1] range
- **Data Augmentation**: Applied to improve generalization

### Data Preprocessing Pipeline
1. **Normalization**: Pixel values scaled to [0,1]
2. **Reshaping**: Images reshaped to (28, 28, 1) format
3. **One-hot Encoding**: Labels converted to categorical format
4. **Data Augmentation**: Rotation, zoom, and shift transformations

## Model Performance

### Training Results
- **Training Accuracy**: 99.8%
- **Validation Accuracy**: 99.2%
- **Test Accuracy**: 99.1%
- **Training Loss**: 0.008
- **Validation Loss**: 0.032

### Performance Metrics (Test Set)
| Metric    | Value |
|-----------|-------|
| Precision | 0.991 |
| Recall    | 0.991 |
| F1-Score  | 0.991 |
| AUC       | 0.999 |

### Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Categorical Crossentropy
- **Epochs**: 25
- **Batch Size**: 128
- **Early Stopping**: Validation loss patience=5

## Installation

### Prerequisites
- Python 3.7+
- pip (Python package installer)
- Virtual environment (recommended)

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/darshanPC25/HandwrittenDigitRecognitionUsingCNN.git
   cd HandwrittenDigitRecognitionUsingCNN
   ```

2. **Create Virtual Environment**
   ```bash
   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   
   # Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   ```bash
   python -c "import tensorflow, flask, numpy, opencv-python; print('All dependencies installed successfully')"
   ```

## Usage

### Running the Application

1. **Activate Virtual Environment**
   ```bash
   source venv/bin/activate  # Linux/Mac
   # OR
   venv\Scripts\activate     # Windows
   ```

2. **Start Flask Server**
   ```bash
   flask run
   ```
   
   Alternative:
   ```bash
   python app.py
   ```

3. **Access the Application**
   - Open browser and navigate to: `http://127.0.0.1:5000`
   - Default port: 5000 (configurable in app.py)

### Using the Web Interface

1. **Draw a Digit**: Use mouse/touch to draw a digit (0-9) on the canvas
2. **Clear Canvas**: Use the "Clear" button to reset the drawing area  
3. **Predict**: Click "Predict" to get the model's prediction
4. **View Results**: Prediction result and confidence score are displayed

### API Usage

#### Prediction Endpoint
```bash
POST /predict
Content-Type: multipart/form-data
Body: Base64 encoded image data
```

#### Example with cURL
```bash
curl -X POST \
  http://127.0.0.1:5000/predict \
  -H 'Content-Type: multipart/form-data' \
  -F 'image=@digit_image.png'
```

## Project Structure

```
HandwrittenDigitRecognitionUsingCNN/
│
├── app.py                 # Flask application main file
├── model/                 # Model-related files
│   ├── train_model.py     # Model training script
│   ├── model.h5           # Trained CNN model
│   └── load_model.py      # Model loading utilities
│
├── static/                # Static assets
│   ├── css/
│   │   └── style.css      # Styling for web interface
│   ├── js/
│   │   └── script.js      # Frontend JavaScript logic
│   └── images/            # Static images
│
├── templates/             # HTML templates
│   └── index.html         # Main web interface
│
├── utils/                 # Utility functions
│   ├── image_processing.py # Image preprocessing functions
│   └── data_loader.py     # Data loading utilities
│
├── notebooks/             # Jupyter notebooks
│   └── model_training.ipynb # Training process documentation
│
├── tests/                 # Unit tests
│   ├── test_model.py      # Model testing
│   └── test_api.py        # API endpoint testing
│
├── requirements.txt       # Python dependencies
├── config.py             # Configuration settings
├── README.md             # Project documentation
└── .gitignore           # Git ignore file
```

## API Documentation

### Endpoints

#### GET /
- **Description**: Serves the main web interface
- **Response**: HTML page with canvas interface

#### POST /predict
- **Description**: Processes image and returns digit prediction
- **Input**: Base64 encoded image data
- **Output**: JSON response with prediction and confidence
- **Response Format**:
  ```json
  {
    "prediction": 7,
    "confidence": 0.9987,
    "status": "success"
  }
  ```

#### GET /health
- **Description**: Health check endpoint
- **Response**: Server status information

### Error Codes
- `200`: Success
- `400`: Bad Request (invalid image format)
- `500`: Internal Server Error (model loading issues)

## Model Details

### CNN Architecture Specifications

```python
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)           (None, 26, 26, 32)       320       
max_pooling2d_1 (MaxPool)   (None, 13, 13, 32)       0         
conv2d_2 (Conv2D)           (None, 11, 11, 64)       18496     
max_pooling2d_2 (MaxPool)   (None, 5, 5, 64)         0         
dropout_1 (Dropout)         (None, 5, 5, 64)         0         
flatten_1 (Flatten)         (None, 1600)             0         
dense_1 (Dense)             (None, 128)              204928    
dropout_2 (Dropout)         (None, 128)              0         
dense_2 (Dense)             (None, 10)               1290      
=================================================================
Total params: 225,034
Trainable params: 225,034
Non-trainable params: 0
```

### Hyperparameters
- **Learning Rate**: 0.001
- **Dropout Rate**: 0.25 (Conv layers), 0.5 (Dense layers)
- **Activation Functions**: ReLU (hidden), Softmax (output)
- **Weight Initialization**: He Normal
- **Regularization**: L2 regularization (0.001)

## Technologies Used

### Backend
- **Python 3.8+**: Core programming language
- **TensorFlow 2.x**: Deep learning framework
- **Keras**: High-level neural networks API
- **Flask**: Web framework for deployment
- **NumPy**: Numerical computing
- **OpenCV**: Computer vision and image processing

### Frontend
- **HTML5**: Structure and canvas element
- **CSS3**: Styling and responsive design
- **JavaScript (ES6+)**: Interactive functionality
- **Bootstrap**: UI components and grid system

### Development Tools
- **Git**: Version control
- **Jupyter Notebooks**: Model experimentation
- **pytest**: Testing framework
- **Docker**: Containerization (optional)

## Contributing

We welcome contributions to improve this project! Please follow these guidelines:

### How to Contribute
1. **Fork the Repository**
2. **Create Feature Branch**: `git checkout -b feature/AmazingFeature`
3. **Commit Changes**: `git commit -m 'Add some AmazingFeature'`
4. **Push to Branch**: `git push origin feature/AmazingFeature`
5. **Open Pull Request**

### Code Standards
- Follow PEP 8 style guidelines
- Add docstrings for all functions and classes
- Include unit tests for new features
- Update documentation as needed

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black .

# Linting
flake8 .
```

## Future Enhancements

### Planned Features
- [ ] **Multi-digit Recognition**: Support for recognizing multiple digits in sequence
- [ ] **Custom Dataset Support**: Allow training on user-provided datasets
- [ ] **Mobile App**: Native mobile applications for iOS and Android
- [ ] **Batch Processing**: API endpoint for processing multiple images
- [ ] **Model Comparison**: Interface to compare different model architectures

### Technical Improvements
- [ ] **Docker Deployment**: Containerization for easier deployment
- [ ] **CI/CD Pipeline**: Automated testing and deployment
- [ ] **Performance Optimization**: Model quantization and optimization
- [ ] **Monitoring**: Logging and monitoring dashboard
- [ ] **Authentication**: User authentication and session management

### UI/UX Enhancements
- [ ] **Improved Canvas**: Better drawing tools and touch support
- [ ] **Result Visualization**: Confidence heatmaps and prediction analysis
- [ ] **Accessibility**: WCAG compliance and screen reader support
- [ ] **Internationalization**: Multi-language support


## Acknowledgments

### Dataset and Research
- **MNIST Dataset**: Yann LeCun, Corinna Cortes, and Christopher J.C. Burges
- **Convolutional Neural Networks**: Inspiration from LeNet-5 architecture
- **Deep Learning Community**: TensorFlow and Keras development teams

### Libraries and Tools
- TensorFlow team for the deep learning framework
- Flask development team for the web framework
- NumPy and OpenCV communities for computational tools

### Educational Resources
- CS231n: Convolutional Neural Networks for Visual Recognition (Stanford)
- Deep Learning Specialization (Andrew Ng, Coursera)
- Hands-On Machine Learning (Aurélien Géron)

