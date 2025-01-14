import cv2
import numpy as np
import tensorflow as tf

# Load the Teachable Machine model
model_path = "model_unquant.tflite"  # Replace with your model's path
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get model input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use the default webcam (change 0 if needed)

# Define a dictionary for mapping labels to names
label_to_name = {
    0: "Kaveri R.",  # Your name
    1: "Unknown"      # For others, or add more labels if applicable
}


def predict(frame):
    # Preprocess the frame to match the model's input requirements
    img = cv2.resize(frame, (224, 224))  # Adjust size based on your model
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]

    # Run the model
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Find the label with the highest probability
    predicted_label = np.argmax(output_data)
    confidence = np.max(output_data)
    return predicted_label, confidence


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror-like view
    frame = cv2.flip(frame, 1)

    # Predict using the model
    label, confidence = predict(frame)
    name = label_to_name.get(label, "Unknown")  # Map label to name
    text = f"Name: {name}, Confidence: {confidence:.2f}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the video feed
    cv2.imshow("Face Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
