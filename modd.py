import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Define the path to the testing directory
test_data_dir = 'basedata/testing'

# Load your trained model
model = tf.keras.models.load_model('face_recognition_model.h5')

# Iterate through each image in the testing directory
for image_file_name in os.listdir(test_data_dir):
    # Load and preprocess the image
    img_path = os.path.join(test_data_dir, image_file_name)
    img = image.load_img(img_path, target_size=(200, 200))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # Rescale pixel values to [0, 1]

    # Use the model to make predictions
    predictions = model.predict(x)
    predicted_class_index = np.argmax(predictions)
    confidence_score = predictions[0][predicted_class_index]

    # Get the predicted class label
    class_labels = ['arvind', 'prashanth']
    predicted_class_label = class_labels[predicted_class_index]

    # Display the image file name, predicted class, and confidence score
    print(f'Image: {image_file_name}, Predicted Class: {predicted_class_label}, Confidence: {confidence_score:.2f}')

