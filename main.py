import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import argparse
import cv2

# Set model path
MODEL_PATH = '/home/akash/shristi_model/Shristi.keras'

# Load the model
print("Loading waste classification model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# Class names (in the same order as during training)
class_names = ['cardboard', 'compost', 'glass', 'metal', 'paper', 'plastic', 'trash']

def preprocess_image(img):
    """Preprocess image for prediction"""
    # Resize to model's expected size
    img = cv2.resize(img, (224, 224))
    
    # Convert to array and normalize
    img_array = np.array(img)
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(img):
    """Predict waste category for an image"""
    # Preprocess
    processed_img = preprocess_image(img)
    
    # Predict
    prediction = model.predict(processed_img, verbose=0)
    
    # Get class and confidence
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class] * 100
    
    return class_names[predicted_class], confidence, prediction[0]

def display_prediction(img, category, confidence, all_probs):
    """Display image with prediction overlay"""
    # Create a copy of the image
    result_img = img.copy()
    
    # Colors for different waste types (BGR format)
    colors = {
        'cardboard': (42, 42, 165),    # Brown
        'compost': (0, 128, 0),        # Green
        'glass': (230, 216, 173),      # Light Blue
        'metal': (192, 192, 192),      # Silver
        'paper': (255, 255, 0),        # Yellow
        'plastic': (0, 0, 255),        # Red
        'trash': (128, 128, 128)       # Gray
    }
    
    # Add prediction text
    color = colors.get(category, (0, 0, 0))
    cv2.putText(result_img, f"{category.upper()}: {confidence:.1f}%", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Display each category probability as a bar
    bar_width = 150
    for i, prob in enumerate(all_probs):
        # Draw category name
        cv2.putText(result_img, f"{class_names[i]}", 
                    (10, 70 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw probability bar
        bar_length = int(prob * bar_width)
        cv2.rectangle(result_img, (110, 60 + i*30), (110 + bar_length, 75 + i*30), 
                     colors.get(class_names[i], (0, 0, 0)), -1)
        
        # Add percentage
        cv2.putText(result_img, f"{prob*100:.1f}%", 
                    (270, 70 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return result_img

def process_webcam():
    """Use webcam for real-time waste classification"""
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Webcam activated. Press 'q' to quit or close the window to exit.")
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Make prediction
            category, confidence, all_probs = predict_image(frame)
            
            # Display prediction on frame
            result_frame = display_prediction(frame, category, confidence, all_probs)
            
            # Display the frame
            cv2.imshow('Waste Classification', result_frame)
            
            # Check for 'q' key or window close
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or cv2.getWindowProperty('Waste Classification', cv2.WND_PROP_VISIBLE) < 1:
                break
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam deactivated.")

def process_image(image_path):
    """Process a single image file"""
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} not found")
        return
    
    # Read image
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # Make prediction
    category, confidence, all_probs = predict_image(img)
    
    # Print results
    print(f"\nPrediction for {os.path.basename(image_path)}:")
    print(f"Category: {category}")
    print(f"Confidence: {confidence:.2f}%")
    
    # Display prediction on image
    result_img = display_prediction(img, category, confidence, all_probs)
    
    # Display the image
    cv2.imshow('Waste Classification', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Waste Classification')
    parser.add_argument('--image', type=str, help='Path to image file')
    args = parser.parse_args()
    
    if args.image:
        process_image(args.image)
    else:
        process_webcam() 
