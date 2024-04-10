import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras import layers, models, datasets
from PIL import Image
import numpy as np

# Loading the trained model
model = tf.keras.models.load_model('mnist_model.keras')

app = Flask(__name__)

# route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve the image file from the request
    image_file = request.files['image']

    # Convert the image file to a NumPy array
    image = Image.open(image_file).convert('L') 
    image = image.resize((28, 28))  
    image = np.array(image)
    image = 255 - image
    image = image / 255.0  

    prediction = model.predict(np.expand_dims(image, axis=0))

    # Get the predicted class label
    predicted_class = np.argmax(prediction)

    return jsonify({'predicted_class': int(predicted_class)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)