import tensorflow as tf
import os

def get_file_size(file_path):
    size = os.path.getsize(file_path)
    return size / 1024  # Size in KB

print("[INFO] Loading full model...")
model = tf.keras.models.load_model('railway_model_full.h5')

# 1. CONVERT TO TFLITE (Standard)
# -------------------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('railway_model.tflite', 'wb') as f:
    f.write(tflite_model)

# 2. CONVERT TO TFLITE (Quantized - The TinyML Magic)
# ---------------------------------------------------
# This converts Float32 -> Int8 (4x smaller!)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model_quant = converter.convert()

with open('railway_model_quantized.tflite', 'wb') as f:
    f.write(tflite_model_quant)

# 3. COMPARE SIZES (The Portfolio "Money Shot")
# ---------------------------------------------
base_size = get_file_size('railway_model_full.h5')
quant_size = get_file_size('railway_model_quantized.tflite')

print("\n" + "="*40)
print(f"Original Model Size:   {base_size:.2f} KB")
print(f"Quantized Model Size:  {quant_size:.2f} KB")
print(f"REDUCTION:             {100 - (quant_size/base_size)*100:.1f}%")
print("="*40 + "\n")
print("[INFO] This proves the model fits on ESP32/Edge Hardware!")
