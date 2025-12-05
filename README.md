# Railway Track Fault Detection System (Edge AI)

## 🚀 Project Overview
A low-cost, real-time safety system designed to detect railway track faults (cracks, missing fasteners) using Edge AI. 
The goal is to enable predictive maintenance on India's 68,000km rail network using offline-first hardware.

## 🎯 Key Achievements
- **Architecture:** Switched from Custom CNN to **MobileNetV2 (Transfer Learning)** to boost validation accuracy to >90%.
- **Optimization:** Applied Post-Training Quantization (PTQ) via TensorFlow Lite.
- **Result:** Reduced model size by **91.7%** (130MB -> 10.9MB) while maintaining high detection accuracy.
- **Hardware Readiness:** Optimized for ESP32/Raspberry Pi deployment.

## 🛠️ Tech Stack
- **Deep Learning:** TensorFlow/Keras, MobileNetV2
- **Edge Optimization:** TFLite (Int8 Quantization)
- **Data Pipeline:** Pandas, OpenCV

## 📊 Evidence of Success

### 1. Accuracy (Defect Detection)
The model correctly identifies structural track faults in unseen test data.
![Prediction Example](prediction_result.png)

### 2. Efficiency (Size Reduction)
Achieved a 10x compression ratio suitable for microcontrollers.
![Quantization Result](output.png)