# 🧠 K-Nearest Neighbors & Softmax Classifiers for CIFAR-10  

This project implements **K-Nearest Nearest Neighbors (KNN) and Softmax classifiers** for **image classification** on the **CIFAR-10 dataset**. It includes **hyperparameter tuning, k-fold cross-validation, vectorization**, and **Stochastic Gradient Descent (SGD) optimization** to improve efficiency and accuracy.

---

## 📂 Project Structure
```plaintext
KNN-Softmax-ML/
│── KNN.ipynb              # Implementation of KNN classifier
│── Softmax.ipynb          # Implementation of Softmax classifier
│── requirements.txt       # Dependencies for running the project
│── README.md              # Project documentation (this file)
└── data/                  # Folder to store CIFAR-10 dataset
```

---

## 🚀 Getting Started

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/yourusername/KNN-Softmax-ML.git
cd KNN-Softmax-ML
```

### **2️⃣ Install Dependencies**
You may need to install required packages before running the notebooks:
```bash
pip install -r requirements.txt
```

### **3️⃣ Run the Jupyter Notebook**
```bash
jupyter notebook
```
Open `KNN.ipynb` or `Softmax.ipynb` to explore the implementations.

---

## 🔍 Project Overview

### **1️⃣ K-Nearest Neighbors (KNN) Classifier**
✅ **Implemented a KNN classifier** to classify images from the CIFAR-10 dataset.  
✅ **Optimized distance calculations** using vectorization for efficiency.  
✅ **Performed hyperparameter tuning** with **k-fold cross-validation** to determine the best **k-value** and distance **norm (L1, L2, Linf)**.  
✅ **Evaluated performance** based on test accuracy and cross-validation scores.  

### **2️⃣ Softmax Classifier**
✅ **Implemented Softmax classification** for multi-class image recognition.  
✅ **Computed gradients and optimized using Stochastic Gradient Descent (SGD)**.  
✅ **Fine-tuned learning rate and hyperparameters** for improved performance.  
✅ **Achieved ~41.3% validation accuracy on CIFAR-10 dataset**.

---

## 📊 Results Summary
- **Best KNN Model:** k = 10, using L1 norm (best cross-validation error ~0.69).
- **Best Softmax Model:** Achieved **~41.3% accuracy** after optimizing hyperparameters.
- **Performance Comparison:** KNN struggled with large data sizes, while Softmax improved accuracy with SGD tuning.

---

## 📌 Additional Resources
- 📄 **Jupyter Notebook Viewer**:
  - [nbviewer](https://nbviewer.jupyter.org/github/cindydingg/KNN-Softmax-ML/blob/main/knn_nosol.ipynb)
  - [nbviewer](https://nbviewer.jupyter.org/github/cindydingg/KNN-Softmax-ML/blob/main/softmax_nosol.ipynb)
- 🚀 **Google Colab Links**:
  - [KNN.ipynb](https://colab.research.google.com/github/cindydingg/KNN-Softmax-ML/blob/main/knn_nosol.ipynb)
  - [Softmax.ipynb](https://colab.research.google.com/github/cindydingg/KNN-Softmax-ML/blob/main/softmax_nosol.ipynb)

---

## 🤝 Contributing
If you would like to contribute, feel free to fork this repository, make changes, and submit a pull request.

---

## 📜 License
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

### 🌟 **If you find this project useful, please ⭐ the repo!**
