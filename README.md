🧠 PyTorch CNN Classification

A comprehensive PyTorch course for Convolutional Neural Networks (CNN) and image classification.

🔗 Open In Colab: https://colab.research.google.com/github/Faeze-OstadHoseini/Pytorch-CNN-Classification


📚 About The Project

This project is a complete educational course for learning PyTorch and Convolutional Neural Networks (CNN). From basic concepts to implementing advanced models like LeNet-5, everything is taught step by step using Jupyter Notebook.

🎯 Learning Objectives

✅ Understanding PyTorch basics (Tensors, Autograd, Datasets)
✅ Implementing linear regression from scratch
✅ Building Convolutional Neural Networks (CNN)
✅ Implementing LeNet-5 architecture
✅ Image classification with PyTorch
✅ Using DataLoader, Sampler, and Transforms
✅ Using TensorBoard for monitoring
✅ Working with Docker for development environment


📁 Project Structure
```text
Pytorch-CNN-Classification/
│
├── 📓 01-01.ipynb              # PyTorch Introduction
├── 📓 01-02.ipynb              # Basic Concepts Continued
├── 📓 01-1.py                  # Base Script
├── 📓 Chapter02.ipynb          # Linear Regression with PyTorch
├── 📓 Chapter02.1.ipynb        # Regression Continued
├── 📓 Chapter03.ipynb          # Logistic Regression
├── 📓 Chapter04.ipynb          # Convolutional Neural Networks (CNN)
├── 📓 Chapter05.ipynb          # LeNet-5 and Classification
├── 📓 Chaptor06.ipynb          # Advanced Topics
├── 📓 Chaptor07.ipynb          # Final Project
│
├── 🐳 docker-compose.yml       # Docker Compose for GPU execution
├── 📦 environment.yml          # Conda environment
├── 📄 .gitignore               # Git ignored files
├── 🔧 helpers.py               # Helper functions
│
├── 📂 data_generation/         # Synthetic data generation
│   ├── image_classification.py # Data for image classification
│   └── simple_linear_regression.py # Data for regression
│
├── 📂 data_preparetion/        # Data preparation
│   ├── v0.py                   # Version 0 - Convert to tensor
│   ├── v1.py                   # Version 1 - Add DataLoader
│   └── v2.py                   # Version 2 - Train/validation split
│
├── 📂 model_configuration/     # Model configuration
│   ├── v0.py                   # Version 0 - Base model
│   ├── v1.py                   # Version 1 - Add train_step
│   ├── v2.py                   # Version 2 - Add val_step
│   ├── v3.py                   # Version 3 - Add TensorBoard
│   └── v4.py                   # Version 4 - Final model
│
├── 📂 model_training/          # Training loops
│   ├── v0.py                   # Version 0 - Simple loop
│   ├── v1.py                   # Version 1 - With train_step
│   ├── v2.py                   # Version 2 - With batch
│   ├── v3.py                   # Version 3 - With mini_batch
│   ├── v4.py                   # Version 4 - With validation
│   └── v5.py                   # Version 5 - With TensorBoard
│
└── 📂 stepbystep/              # StepByStep class
    ├── v0.py                   # Version 0 - Base class
    └── v1.py                   # Version 1 - Complete class
```

🚀 Installation and Setup

Method 1: Using Docker (Recommended for GPU)

# 1. Clone the project
git clone https://github.com/your-username/Pytorch-CNN-Classification.git
cd Pytorch-CNN-Classification

# 2. Run with Docker Compose
docker-compose up -d

# 3. Open Jupyter
# In browser: http://localhost:8888

Method 2: Using Conda

# 1. Create Conda environment
conda env create -f environment.yml

# 2. Activate environment
conda activate pytorch-env

# 3. Run Jupyter
jupyter notebook

Method 3: Using pip

# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)

# 2. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install jupyter numpy matplotlib scikit-learn tensorboard

# 3. Run Jupyter
jupyter notebook


📊 Course Content

Chapter 1: PyTorch Introduction
- Introduction to Tensors
- Autograd and Gradient Computation
- Building Simple Models

Chapter 2: Linear Regression
- Implementation from Scratch with Numpy
- Implementation with PyTorch
- Using DataLoader

Chapter 3: Logistic Regression
- Binary Classification
- BCELoss Function
- Model Evaluation

Chapter 4: Convolutional Neural Networks (CNN)
- Introduction to Conv2d
- MaxPooling
- Building a Simple CNN

Chapter 5: LeNet-5
- LeNet-5 Architecture
- MNIST Classification
- Performance Improvement

Chapter 6: Advanced Topics
- Dropout
- Batch Normalization
- Data Augmentation

Chapter 7: Final Project
- Practical Image Classification Project
- Hyperparameter Optimization
- Model Saving and Loading


🛠️ Technologies Used

PyTorch        Deep learning framework
Jupyter Notebook  Interactive learning environment
Docker         Development environment containerization
TensorBoard    Monitoring and visualization
CUDA           GPU support
Matplotlib     Plotting and image display
NumPy          Numerical computations


🤝 Contributing

If you'd like to contribute to this project:

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

📧 Contact

GitHub: github.com/Faeze-OstadHoseini
Email: ostadhoseinifaeze@gmail.com


⭐ If you found this project helpful, please give it a star!

Made with ❤️ by Faeze OstadHoseini
