# Autonomous Automotive Safety: Real-Time Driver Drowsiness Detection System

An edge-optimized, deep learning computer vision framework engineered for real-time driver drowsiness detection. This system utilizes a customized **MobileNetV3** convolutional neural network architecture trained on the **Driver Drowsiness Dataset (DDD)** to achieve highly accurate, low-latency classification of driver fatigue states under variable cabin lighting conditions.

---

## Project Overview

Driver fatigue is a leading contributor to automotive accidents globally. Traditional facial landmark tracking systems often suffer from high computational overhead, rendering them impractical for resource-constrained embedded automotive hardware. 

This project addresses this limitation by developing a lightweight, high-throughput binary classifier that:
1. **Monitors Driver State:** Processes real-time video streams to evaluate facial structures and eye closure metrics.
2. **Optimizes Edge Deployment:** Leverages MobileNetV3 to guarantee minimal inference latency and low power consumption without sacrificing accuracy.

---

## System Architecture & Pipeline

The system operates in a streamlined, sequential pipeline designed to move from raw visual capture to instant safety alerts:

`Video Stream Ingestion ──> Frame Preprocessing ──> MobileNetV3 Core Inference ──> State Classification ──> Alarm Trigger Loop`

### 1. Frame Ingestion & Preprocessing
* Captures high-frame-rate video input simulating an in-cabin dashboard camera.
* Standardizes incoming frames via normalization, resizing, and pixel scaling to match the strict mathematical input dimensions required by the deep learning model.

### 2. MobileNetV3 Feature Extraction
* Utilizes a fine-tuned **MobileNetV3-Small** backbone, capitalizing on its depthwise separable convolutions and hardware-aware automated search design.
* Extracts complex spatial hierarchies and facial fatigue indicators (such as micro-expressions, head tilt, and prolonged eye closure) while maintaining a drastically minimized parameter count.

### 3. Binary Classification & State Evaluation
* Passes the optimized feature vectors through a dense classification head to output a probability score mapping to two explicit states: `Alert` or `Drowsy`.
* Employs an evaluation threshold to prevent false positives from natural blinking, triggering the safety alarm mechanism only when sustained fatigue patterns are verified.

---

## Dataset & Model Benchmarks

* **Dataset:** Trained and validated using the comprehensive **Driver Drowsiness Dataset (DDD)**, featuring thousands of balanced, high-resolution closed/open eye and fatigue-state facial crops.
* **Architecture Selection:** MobileNetV3 was deliberately selected over heavier architectures (like ResNet or VGG) to hit the strict low-latency thresholds required for real-time collision prevention systems.

---

## Installation & Setup

### Prerequisites
* Python 3.10 or higher
* Recommended virtual environment manager (`venv`)

### 1. Clone the Repository
```bash
git clone [https://github.com/LiamBF/driver-drowsiness-detection.git](https://github.com/LiamBF/driver-drowsiness-detection.git)
cd driver-drowsiness-detection
