from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow.keras as keras
import os

app = Flask(__name__)

# Load the Keras model and weights
model = keras.models.load_model('model/my_model')
model.load_weights('model/best_weights.h5')

# Define the class labels
label_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((48, 48))  # Resize the image to the input size of the model
    img = img.convert('L')  # Convert image to grayscale
    img = np.array(img)  # Convert image to numpy array
    img = np.expand_dims(img, axis=0)  # Add an extra dimension for batch size
    img = img / 255.0  # Normalize the image
    return img

# Function to make predictions
def make_prediction(image):
    preprocessed_img = preprocess_image(image)
    prediction = model.predict(preprocessed_img)
    predicted_class = np.argmax(prediction)
    class_label = label_dict[predicted_class]
    return class_label

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file:
    # Read the image file
            image_path = os.path.join('static', uploaded_file.filename)
            uploaded_file.save(image_path)
            img = Image.open(image_path)

            # Make prediction on the uploaded image
            prediction = make_prediction(img)

            # Display the predicted class label
            print("Predicted Emotion:", prediction)
            return render_template('result.html', prediction=prediction, image_path=image_path)

    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)
