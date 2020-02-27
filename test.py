import tensorflow as tf
from utils import preprocess_test_image
import numpy as np
import pathlib

# Loading the list of Class Names for prediction
path = r"C:\Users\shiva\Deepfont\train"
data = pathlib.Path(path)
Class_Names = np.array([item.name for item in data.glob('*') if item.name != "LICENSE.txt"])

# Loading and pre-processing the test image from the given path
test_path = r"C:\Users\shiva\Deepfont\test\advisorily_157.jpg"
img = preprocess_test_image(test_path)

# Loading the pre-trained saved model from the given path
model_path = r'C:\Users\shiva\Deepfont\CNN_Font_Classification.h5'
model = tf.keras.models.load_model(model_path)

# Predicting the font class from the given single image
prediction = model.predict_classes(img)
print('Predicted Class: {}'.format(Class_Names[prediction]))
