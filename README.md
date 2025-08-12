# Handwritten Digit Recognition using PyTorch & Logistic Regression

This project implements a simple **Optical Character Recognition (OCR)** system using **Logistic Regression** in **PyTorch** to classify handwritten digits from the MNIST dataset (0–9).  

The aim is to demonstrate that even a simple linear model can achieve good performance on OCR tasks with proper **data preprocessing**, **model training**, and **evaluation**.

---

## 📌 Features
- Custom Logistic Regression model built using `torch.nn.Module`.
- Trains on the **MNIST** handwritten digit dataset.
- Step-by-step approach: data preprocessing → feature extraction → model training → evaluation.
- Achieves ~90–92% accuracy on MNIST.
- Lightweight and cost-effective OCR baseline.
- Written with **PyTorch** for easy extension to more complex models.

---

## 📂 Project Structure
├── src/
│ ├── model.py # Logistic regression model definition
│ ├── train.py # Training loop
│ ├── evaluate.py # Model evaluation
│ ├── predict.py # Single image prediction
│ ├── utils.py # Helper functions & data loading
├── requirements.txt
├── README.md
└── LICENSE


---

## ⚙️ Requirements
- Python 3.8+
- PyTorch & torchvision
- NumPy
- Matplotlib (optional, for plotting)
- scikit-learn (optional, for metrics)

Install dependencies:
```bash
pip install -r requirements.txt

Example requirements.txt:

nginx
Copy
Edit
torch
torchvision
numpy
matplotlib
scikit-learn
tqdm
🚀 Training
python src/train.py --epochs 20 --batch-size 128 --lr 0.1 --device cuda

📈 Evaluation
python src/evaluate.py --checkpoint checkpoints/logreg_mnist.pt --device cpu

🔍 Prediction
python src/predict.py --image path/to/digit.png --checkpoint checkpoints/logreg_mnist.pt

