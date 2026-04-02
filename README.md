# 🚗 Car Detection on BDD100K using Faster R-CNN

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Computer Vision](https://img.shields.io/badge/CV-Object%20Detection-green)

An end-to-end computer vision pipeline to train, validate, and test a deep learning model for detecting cars in the BDD100K dataset. This project leverages **PyTorch** and a mobile-optimized **Faster R-CNN** architecture to achieve high-fidelity predictions on a resource-constrained compute budget.

> 🚧 **Work in Progress:** This project is actively being developed. Future updates include data augmentation upgrades, multi-class detection, and performance benchmarking.

---

## 🌟 Key Features
* **Production-Ready Architecture:** Refactored from a single script into clean, modular files (`dataset`, `train`, `inference`).
* **Optimized for Speed:** Uses a `fasterrcnn_mobilenet_v3_large_320_fpn` backbone for rapid inference and lower VRAM consumption.
* **Mixed-Precision Training:** Implements PyTorch's `torch.cuda.amp` to speed up training while saving GPU memory.
* **Dynamic Scaling:** Custom dataset loader that dynamically resizes images and scales bounding box coordinates proportionally.
* **No Hardcoded Paths:** Leverages Python's `argparse` making the scripts highly shareable and easy to run on any machine.
* **Manageable for 4GB VRAM GPU.

---

## 📊 Results & Demo
*(Tip for the author: Once you have training complete, replace this with a screenshot or a GIF of your model detecting cars!)*

Green boxes represent the model's predictions, while red boxes represent the ground truth labels from the BDD100K dataset.

---

## 🛠️ Tech Stack
* **Framework:** PyTorch & Torchvision
* **Data Manipulation:** NumPy, JSON, PIL
* **Visualization:** OpenCV, Matplotlib
* **Metrics:** Torchmetrics (Mean Average Precision - mAP)

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone [https://github.com/your-username/car-detection-bdd100k.git](https://github.com/your-username/car-detection-bdd100k.git)
cd car-detection-bdd100k
pip install -r requirements.txt

### 2. Training the Model
To train the model on your BDD100k subset, run:
```bash
python train.py --img_dir "path/to/images" --json_dir "path/to/labels" --num_images 1000 --epochs 15

## 🗺️ Roadmap / Future Improvements
To take this project to a production/automotive grade, I am planning to implement:

[ ] Torchvision V2 Transforms: To handle automatic bounding box scaling during complex augmentations (like random cropping and flipping).

[ ] Multi-Class Support: Expanding detection from just car to pedestrians, traffic lights, and riders.

[ ] Inference Optimization: Exporting the trained PyTorch model to ONNX or TensorRT for deployment on edge devices.
