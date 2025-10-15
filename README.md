# Real-Time-Sign-Language-Detection
Real-time Sign Language Detection using Python,Tensorflow,Deeplearning OpenCV, and MediaPipe to translate hand gestures into text.
 ## Sign Language Detection System##
 

A real-time sign language detection system using Machine Learning, Computer Vision, and Deep Learning. This project uses MediaPipe for hand landmark detection and TensorFlow for gesture classification.
Show Image
Show Image
Show Image
Show Image
 
 # Table of Contents

Features
Demo
Technologies Used
Installation
Usage
Project Structure
Model Architecture
How It Works
Troubleshooting
Contributing
Future Enhancements
License
Contact

 # Features

 1. Real-time Detection: Instant gesture recognition through webcam
 2. Custom ML Model: Train your own neural network with custom gestures
️ 3. Hand Landmark Detection: Accurate hand tracking using MediaPipe
 4. Model Persistence: Save and load trained models
 5. Multiple Data Sources: Collect data from webcam, images, or pre-saved datasets
 6. High Accuracy: Achieves 90%+ accuracy with proper training data 
7. Easy to Extend: Add new gestures by simply collecting more data

# Demo

Real-time Detection
The system detects hand gestures in real-time and displays the predicted sign with confidence score.
Supported Gestures (Default):

A, B, C, D, E, F, G, H, I, J (expandable to full alphabet)

# Technologies Used
TechnologyPurposeVersionPythonProgramming Language3.8+TensorFlow/KerasDeep Learning Framework2.15.0MediaPipeHand Landmark Detection0.10.9OpenCVComputer Vision & Camera4.8.1.78NumPyNumerical Computing1.26.4
Why These Libraries?

TensorFlow: Creates and trains the neural network model for gesture classification
MediaPipe: Detects 21 hand landmarks in real-time with high accuracy
OpenCV: Handles camera input/output and image processing
NumPy: Efficient array operations and data normalization

# Installation
Prerequisites

Python 3.8 or higher
Webcam/Camera
Windows/Linux/MacOS

Step 1: Clone the Repository
bashgit clone https://github.com/YOUR_USERNAME/sign-language-detection.git
cd sign-language-detection
Step 2: Create Virtual Environment (Recommended)
bash# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
Step 3: Install Dependencies
bashpip install -r requirements.txt
Requirements
tensorflow==2.15.0
mediapipe==0.10.9
opencv-python==4.8.1.78
numpy==1.26.4
protobuf==4.23.4
💻 Usage
Option 1: Train Your Own Model (Recommended for First Time)
bashpython sign_language_detection.py

Choose Option 1 - Collect training data from webcam
The program will guide you through collecting samples for each gesture
Show the gesture (e.g., "A") to the camera
Press SPACE to start collecting samples
Keep your hand steady while 100 samples are collected
Repeat for all gestures
Model trains automatically after data collection
Start real-time detection when prompted

Option 2: Load Pre-trained Model
bashpython sign_language_detection.py

Choose Option 4 - Load existing model and start detection
Show gestures to the camera
See real-time predictions with confidence scores

Option 3: Train from Image Dataset
Folder Structure:
dataset/
  ├── A/
  │   ├── img001.jpg
  │   ├── img002.jpg
  │   └── ...
  ├── B/
  │   ├── img001.jpg
  │   ├── img002.jpg
  │   └── ...
  └── ...
bashpython sign_language_detection.py

Choose Option 2 - Load dataset from images folder
Enter the path to your dataset folder
Model processes images and trains automatically

Option 4: Load Pre-saved Dataset
bashpython sign_language_detection.py

Choose Option 3 - Load saved dataset file
Enter path to .npz file (e.g., training_data.npz)
Model trains on the loaded data

🎮 Keyboard Controls
KeyActionSPACEStart collecting samples for current gestureSSkip current gesture during data collectionQQuit/Exit the program
📁 Project Structure
sign-language-detection/
│
├── sign_language_detection.py    # Main program file
├── test_camera.py                 # Camera testing utility
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── LICENSE                        # MIT License
├── .gitignore                     # Git ignore rules
│
├── sign_model.h5                  # Trained model (generated after training)
├── labels.pkl                     # Gesture labels (generated after training)
└── training_data.npz              # Saved dataset (optional)
🧠 Model Architecture
Our neural network consists of:
Input Layer (63 features)
    ↓
Dense Layer (128 neurons, ReLU)
    ↓
Dropout (30%)
    ↓
Dense Layer (64 neurons, ReLU)
    ↓
Dropout (30%)
    ↓
Dense Layer (32 neurons, ReLU)
    ↓
Output Layer (Softmax, N classes)
Input Features

21 hand landmarks detected by MediaPipe
Each landmark has 3 coordinates (x, y, z)
Total: 63 features per gesture

Training Parameters

Optimizer: Adam
Loss Function: Sparse Categorical Crossentropy
Epochs: 50 (default)
Batch Size: 32
Validation Split: 20%

🔍 How It Works
1. Hand Detection
pythonMediaPipe → Detects hand in video frame → Extracts 21 landmarks
2. Feature Extraction
python21 landmarks × 3 coordinates = 63 features → Normalized to [0, 1]
3. Prediction
pythonNeural Network → Processes 63 features → Outputs probability for each gesture
4. Display Result
pythonHighest probability gesture → Displayed with confidence score
📊 Performance Tips
For Better Accuracy:
✅ Good Lighting: Ensure adequate lighting in the room
✅ Clear Background: Use a plain background without clutter
✅ Hand Position: Keep hand clearly visible, not too close or far
✅ Consistent Gestures: Maintain consistent gesture shapes during training
✅ More Samples: Collect 200-300 samples per gesture for better accuracy
✅ Variety: Slightly vary hand position and angle during collection
Optimal Camera Setup:

Distance: 2-3 feet from camera
Position: Center of frame
Lighting: Front-facing light source
Background: Solid color, preferably not skin tone

🐛 Troubleshooting
Camera Not Working
Problem: ERROR: Cannot access camera!
Solutions:

Close other applications using the camera (Zoom, Skype, Teams)
Check camera permissions in system settings
Try different camera index:

python   cap = cv2.VideoCapture(1)  # Try 1, 2, 3 instead of 0

Run the camera test utility:

bash   python test_camera.py
Installation Errors
Problem: Package installation fails
Solutions:

Upgrade pip:

bash   pip install --upgrade pip

Install packages individually:

bash   pip install tensorflow==2.15.0
   pip install mediapipe==0.10.9
   pip install opencv-python

Use Python 3.9 or 3.10 (most stable for TensorFlow)

Low Accuracy
Problem: Model predictions are incorrect
Solutions:

Collect more training samples (200+ per gesture)
Ensure consistent gesture shapes during training
Use better lighting conditions
Retrain the model with varied hand positions
Increase training epochs to 100

Model File Too Large for GitHub
Problem: Cannot upload sign_model.h5 (> 100MB)
Solutions:

Add to .gitignore (already included)
Use Git LFS (Large File Storage):

bash   git lfs install
   git lfs track "*.h5"

Share model via Google Drive or Dropbox

🤝 Contributing
Contributions are welcome! Here's how you can help:

Fork the repository
Create a feature branch:

bash   git checkout -b feature/AmazingFeature

Commit your changes:

bash   git commit -m 'Add some AmazingFeature'

Push to the branch:

bash   git push origin feature/AmazingFeature

Open a Pull Request

Areas for Contribution:

Add more sign language gestures
Improve model architecture
Create web interface
Add support for dynamic gestures
Optimize performance
Write better documentation
Fix bugs

🚀 Future Enhancements

 Full Alphabet Support (A-Z, 0-9)
 Dynamic Gestures (words, phrases, sentences)
 Multi-hand Recognition (two-handed signs)
 Mobile App (Android/iOS using T
