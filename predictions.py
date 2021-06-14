# import the necessary packages
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class_labels = [
    "people",
    "vehicle"
]

# load the trained model from disk
print("[INFO] loading model...")
model = load_model('model/my_model')

# Load an image file to test, resizing it to 32x32 pixels (as required by this model)
img = image.load_img("car.png", target_size=(32, 32))

# Convert the image to a numpy array
image_to_test = image.img_to_array(img) / 255

# Add a fourth dimension to the image (since Keras expects a list of images, not a single image)
list_of_images = np.expand_dims(image_to_test, axis=0)

# Make a prediction using the model
results = model.predict(list_of_images)

# Since we are only testing one image, we only need to check the first result
single_result = results[0]

# We will get a likelihood score for all 10 possible classes. Find out which class had the highest score.
most_likely_class_index = int(np.argmax(single_result))
class_likelihood = single_result[most_likely_class_index]

# Get the name of the most likely class
class_label = class_labels[most_likely_class_index]

plt.imshow(image_to_test)
plt.title("This image is a {} - Likelihood: {:2f}".format(class_label, class_likelihood))
plt.show()

# Print the result
print("This image is a {} - Likelihood: {:2f}".format(class_label, class_likelihood))