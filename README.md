# Autonomous Automotive Safety: Real-Time Driver Drowsiness Detection System

An edge-optimized, deep learning computer vision framework engineered for real-time driver drowsiness detection. This system utilizes a customized **MobileNetV3-Large** convolutional neural network architecture trained on the **Driver Drowsiness Dataset (DDD)** to achieve highly accurate, low-latency classification of driver fatigue states under variable cabin lighting conditions.

---

## Project Overview

Driver fatigue is a leading contributor to automotive accidents globally. Traditional facial landmark tracking systems often suffer from high computational overhead, rendering them impractical for resource-constrained embedded automotive hardware. 

This project addresses this limitation by developing a lightweight, high-throughput binary classifier that:
1. **Monitors Driver State:** Processes real-time video streams to evaluate facial structures and detect signs of driver fatigue.
2. **Optimizes Edge Deployment:** Leverages MobileNetV3-Large to guarantee minimal inference latency and low power consumption without sacrificing accuracy.

---

## System Architecture & Pipeline

The system operates in a streamlined, sequential pipeline designed to move from raw visual capture to instant safety alerts:

`Video Stream Ingestion ──> Frame Preprocessing ──> MobileNetV3 Core Inference ──> State Classification ──> Alarm Trigger Loop`

### 1. Frame Ingestion & Preprocessing
* Captures real-time video input via an OpenCV pipeline simulating an in-cabin dashboard camera.
* Standardizes incoming frames to $224 \times 224$ pixels, applies ImageNet normalization, and scales pixel values to match the strict mathematical input dimensions required by the deep learning model.

### 2. MobileNetV3 Feature Extraction
* Utilizes a fine-tuned **MobileNetV3-Large** backbone, capitalizing on its depthwise separable convolutions and efficient architecture.
* Extracts spatial hierarchies and facial fatigue indicators while maintaining a drastically minimized parameter count (~5 million parameters).

### 3. Binary Classification & State Evaluation
* Passes the optimized feature vectors through a customized dense classification head mapped to two explicit states: `Drowsy` (Label 0) or `Non Drowsy` (Label 1).
* Performs inference entirely locally at the edge, guaranteeing data privacy by eliminating external cloud dependencies.

---

## Dataset & Model Benchmarks

* **Dataset:** Trained and validated using the comprehensive **Driver Drowsiness Dataset (DDD)**, featuring approximately 2.57GB of high-resolution visual data structured into explicit classes.
* **Architecture Selection:** MobileNetV3-Large was deliberately selected over heavier architectures (like ResNet or VGG) to hit the strict low-latency thresholds required for real-time collision prevention systems.
* **Performance Metrics:** Achieved an classification accuracy of approximately **92%** with an end-to-end processing latency profile of **15ms to 25ms**, completely outperforming the 200ms industry safety standard.

---

## Installation & Setup

### Prerequisites
* Python 3.10 or higher
* NVIDIA GPU + CUDA (Highly recommended for training phase)

### 1. Clone the Repository
```bash
git clone https://github.com/LiamBF/driver-drowsiness-detection.git
cd driver-drowsiness-detection
```

### 2. Configure the Local Environment
Initialize an isolated virtual environment and install the required dependencies:

```Bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install torch torchvision opencv-python pillow
```

### 3. Dataset Directory Configuration
Before training, ensure your local dataset directory is structured alphabetically as follows for PyTorch's ImageFolder class:


```Plaintext
DDD/
├── Drowsy/       # Maps to Label 0
└── Non Drowsy/   # Maps to Label 1
```
## Execution Guide
### Step 1: 
Model Training & Fine-Tuning
Execute the PyTorch training script to implement transfer learning with an Adam optimizer and CrossEntropyLoss.

```bash
python train.py
```
Output: Saves the optimized model weights as a serialized .pth file into your local experiments or testing directories (e.g., DDD_Tests/).

### Step 2: 
Real-Time Inference Dashboard
Launch the live, interactive GUI dashboard to monitor inference via your system's webcam.

```bash
python dashboard.py
```
Action: Launches a Tkinter-based interface where users can load their trained .pth weight files, view live camera streams via OpenCV, track processing latency metrics, and view real-time visual alerts (red overlay) instantly upon drowsiness detection.

## License
This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). This strong copyleft framework guarantees that the code remains open for public peer review, academic replication, and non-commercial safety research.

See the [LICENSE](LICENSE) file for complete details.
