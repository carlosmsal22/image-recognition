import numpy as np
import cv2
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.utils import img_to_array

def classify_image(img_path):
    # Load model
    model = ResNet50(weights='imagenet')
    
    # Load and preprocess image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # Predict and return results
    preds = model.predict(x)
    return decode_predictions(preds, top=3)[0]

if __name__ == '__main__':
    result = classify_image('football.jpg')
    print("Predicted:", result)
