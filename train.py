import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# 1. SETUP
IMG_SIZE = (224, 224)
BATCH_SIZE = 16  # Smaller batch size helps smaller computers
EPOCHS = 10      # We will train for 10 rounds
DATA_DIR = 'data'
CSV_PATH = 'data/rails.csv'

print(f"[INFO] Loading Data from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)

# 2. DATA AUGMENTATION (Making the model smarter)
# We twist and flip images so the model learns better
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Load Train Data
train_gen = train_datagen.flow_from_dataframe(
    df[df['data set'] == 'train'],
    directory=DATA_DIR,
    x_col='filepaths',
    y_col='labels',
    target_size=IMG_SIZE,
    class_mode='binary',
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Load Validation Data
valid_gen = val_datagen.flow_from_dataframe(
    df[df['data set'] == 'valid'],
    directory=DATA_DIR,
    x_col='filepaths',
    y_col='labels',
    target_size=IMG_SIZE,
    class_mode='binary',
    batch_size=BATCH_SIZE
)

# 3. BUILD THE "PRO" MODEL (MobileNetV2)
# We download a pre-trained brain from Google
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False # Freeze it so we don't break it

# Add our custom layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# 4. TRAIN
print("[INFO] Starting Transfer Learning with MobileNetV2...")
history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=EPOCHS
)

# 5. SAVE
model.save('railway_model_mobilenet.h5')
print("[SUCCESS] Saved 'railway_model_mobilenet.h5'")