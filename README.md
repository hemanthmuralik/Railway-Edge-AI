# Railway Track Fault Detection System (Edge AI)

## 🚀 Project Overview
A low-cost, real-time safety system designed to detect railway track faults (cracks, missing fasteners) using Edge AI. 
The goal is to enable predictive maintenance on India's 68,000km rail network using offline-first hardware (ESP32/Raspberry Pi).

## 🎯 Key Achievements
- **Architecture:** Custom CNN optimized for 224x224 input.
- **Optimization:** Applied Post-Training Quantization (PTQ) to convert Float32 to Int8.
- **Result:** Reduced model size by **91.7%** (130MB -> 10.9MB) with minimal accuracy loss.
- **Hardware Readiness:** Model fits within the flash memory constraints of standard edge microcontrollers.

## 🛠️ Tech Stack
- Python, TensorFlow/Keras
- TensorFlow Lite (TFLite) for Quantization
- OpenCV for Preprocessing
## 📊 Results

### 1. Model Optimization (The "TinyML" Impact)
By switching from standard Float32 to Int8 quantization, we achieved a **91.7% reduction** in model size, allowing deployment on devices with <16MB Flash memory.

![Quantization Result](output.png)

### 2. Real-World Inference
The model successfully identifies track faults in 224x224 input images.

![Prediction Example](prediction_result.png)