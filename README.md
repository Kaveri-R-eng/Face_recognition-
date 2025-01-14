Face recognition project based on google teachable machine
### Project Report: Python-Based Face Recognition Application Using TensorFlow and OpenCV

#### 1. Introduction

Face recognition has emerged as a transformative technology in computer vision, enabling applications in security, personalization, and user authentication. This project focuses on creating a real-time face recognition application leveraging a pre-trained model exported from Google Teachable Machine. Integration with OpenCV allows for real-time image capture and processing, making the application highly interactive and efficient.

#### 2. Objective

The key objectives of this project are:

- To implement a face recognition system using a pre-trained TensorFlow Lite model.
- To integrate real-time webcam feeds using OpenCV for dynamic face recognition.
- To preprocess input data for compatibility with the pre-trained model and provide accurate predictions.

#### 3. Tools and Technologies

- **Programming Language**: Python
- **Libraries and Frameworks**:
  - TensorFlow Lite for model inference.
  - OpenCV for webcam integration and image preprocessing.
  - NumPy for numerical operations.
- **Hardware**:
  - GPU support (optional) for optimized inference.

#### 4. Methodology

##### 4.1 Model Loading

The TensorFlow Lite model, `model.tflite`, trained using Google Teachable Machine, is loaded using the TensorFlow Lite Interpreter. The interpreter optimizes the model for lightweight and fast inference.

##### 4.2 Class Labels

Labels corresponding to the model’s output are stored in `labels.txt`. Each line in the file represents a class label, allowing the application to map model outputs to meaningful predictions.

##### 4.3 Webcam Integration

- OpenCV’s `VideoCapture` class initializes the webcam for live video feed.
- Frames are displayed in real-time using OpenCV’s `imshow` function.
- The application allows exiting the webcam feed using the Esc key for user convenience.

##### 4.4 Image Preprocessing

- Captured frames are resized to match the input dimensions required by the pre-trained model (e.g., 224x224 pixels).
- Images are normalized to the range [-1, 1] for compatibility with the model’s training setup.

##### 4.5 Prediction and Output

- The TensorFlow Lite Interpreter predicts the class of preprocessed images, providing confidence scores.
- Predictions and their associated confidence scores are displayed in real-time within the application interface.

#### 5. Challenges and Solutions

- **Model Compatibility**: Ensured that the TensorFlow Lite model matched the framework’s inference requirements.
- **Webcam Initialization**: Implemented robust error handling to address issues related to webcam access.
- **Performance Optimization**: Used normalization techniques and efficient model loading for real-time processing.

#### 6. Results
The application successfully:

Captures real-time frames from the webcam.
Preprocesses the frames for model inference.
Predicts and displays class labels with confidence scores in real-time.
Below is an example of the application in action, showcasing a real-time prediction along with its corresponding confidence score:
![image](https://github.com/user-attachments/assets/f419dc2e-64cf-4016-8a05-98b1b2905425)
"Figure 1: Real-time Face Recognition with Class Label and Confidence Score."

#### 7. Future Scope

- **Model Improvements**:
  - Train on a diverse dataset to enhance recognition accuracy.
  - Incorporate multi-class and multi-face detection capabilities.
- **User Interface**:
  - Develop a web-based interface using Flask or Streamlit.
  - Add GUI elements for enhanced user interaction.
- **Cloud Deployment**:
  - Host the application on platforms like AWS or Google Cloud for remote access.
- **Advanced Features**:
  - Include emotion detection and pose estimation.

#### 8. Conclusion

This project highlights the seamless integration of TensorFlow Lite and OpenCV for real-time face recognition tasks. The modular approach ensures scalability, making it a strong foundation for more sophisticated computer vision solutions in the future.

