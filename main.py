import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import onnxruntime

def load_model():
    # Load ONNX model
    model_path = "runs/detect/yolo11n_50epochs/weights/best.onnx"
    session = onnxruntime.InferenceSession(model_path)
    return session

def preprocess_image(image, input_size=(384, 384)):
    # Resize and normalize image
    img = cv2.resize(image, input_size)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.expand_dims(img, axis=0)
    return img

def detect_objects(session, image_path):
    original_image = cv2.imread(image_path)
    input_tensor = preprocess_image(original_image)
    
    # Debug prints
    print("Input tensor shape:", input_tensor.shape)
    print("Input tensor range:", np.min(input_tensor), np.max(input_tensor))
    
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})
    
    # Debug model output
    print("Number of outputs:", len(outputs))
    print("Output shapes:", [out.shape for out in outputs])
    print("Output sample:", outputs[0][0][:5])  # First 5 predictions
    
    return original_image

def select_random_test_image():
    
    # Define the path to test images
    test_images_path = "dataset/images/test"
    
    # Get list of all images in the directory
    image_files = [f for f in os.listdir(test_images_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Select a random image
    if not image_files:
        raise ValueError("No images found in test directory")
    
    random_image = random.choice(image_files)
    
    # Return the full path to the selected image
    return os.path.join(test_images_path, random_image)

def main():
    # Load model
    session = load_model()
    
    # Select random test image
    image_path = select_random_test_image()
    
    # Perform detection
    result_image = detect_objects(session, image_path)
    
    # Display result
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()