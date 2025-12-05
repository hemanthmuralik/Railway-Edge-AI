import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os
import random

# Load the new model
model_path = 'railway_model_mobilenet.h5'
print(f"[INFO] Loading {model_path}...")
model = tf.keras.models.load_model('railway_model_mobilenet.h5')

# Pick a random image from the TEST set (not train/valid)
test_dir = 'data/test' 
# Choose a class to test: 'Defective' or 'Non defective'
class_name = 'Defective' 
img_folder = os.path.join(test_dir, class_name)
random_img = random.choice(os.listdir(img_folder))
img_path = os.path.join(img_folder, random_img)

print(f"[INFO] Testing image: {img_path}")

# Preprocess
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict
prediction = model.predict(img_array)
score = prediction[0][0]

# Logic: Sigmoid outputs 0 to 1.
# Assuming 'Defective' is 0 and 'Non-defective' is 1 (Check your train_gen.class_indices to be sure!)
# Usually flow_from_dataframe maps alphabetically: Defective=0, Non=1.
if score < 0.5:
    label = "DEFECTIVE"
    conf = (1 - score) * 100
else:
    label = "NON-DEFECTIVE"
    conf = score * 100

print(f"✅ Prediction: {label} ({conf:.2f}%)")
print(f"❌ Actual: {class_name.upper()}")

# Show image
plt.imshow(img)
plt.title(f"Pred: {label} | Actual: {class_name}")
plt.axis('off')
plt.show()