import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# 1. SETUP & CONFIGURATION
# -----------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = 'data' # Folder where your images are
CSV_PATH = 'data/rails.csv'

print(f"[INFO] Loading data from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)

# 2. DATA LOADERS (The Product Manager's Pipeline)
# -----------------------------------------------
# We use flow_from_dataframe because we have a CSV map
datagen = ImageDataGenerator(rescale=1./255) # Normalize pixels 0-1

train_df = df[df['data set'] == 'train']
valid_df = df[df['data set'] == 'valid']

train_gen = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=DATA_DIR,
    x_col='filepaths', # Check your CSV column name! Might be 'filepaths' or 'image_path'
    y_col='labels',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary', # Defective vs Non-defective
    shuffle=True
)

valid_gen = datagen.flow_from_dataframe(
    dataframe=valid_df,
    directory=DATA_DIR,
    x_col='filepaths',
    y_col='labels',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# 3. BUILD THE "EDGE-OPTIMIZED" MODEL
# -----------------------------------
# We don't use ResNet50 (too big). We build a custom sequential model.
model = Sequential([
    # Layer 1: Feature Extraction
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    
    # Layer 2
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # Layer 3
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # Layer 4: Flatten & Decision
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5), # Prevents overfitting on small datasets
    Dense(1, activation='sigmoid') # Binary output (0 or 1)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. TRAIN
# --------
print("[INFO] Starting training...")
history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=EPOCHS
)

# 5. SAVE THE "BIG" MODEL
# -----------------------
model.save('railway_model_full.h5')
print("[SUCCESS] Model trained and saved as 'railway_model_full.h5'")
