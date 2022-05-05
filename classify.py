from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import numpy as np

from keras.models import load_model

model = load_model('model_saved.h5')

image = load_img('v_data/test/cars/18.jpg', target_size=(224, 224))
img = np.array(image)
img = img / 255.0
img = img.reshape(1,224,224,3)

predictions = (model.predict(img) > 0.5).astype("int32")

print("Predicted Class (0 - Cars , 1- Planes): ", predictions)
