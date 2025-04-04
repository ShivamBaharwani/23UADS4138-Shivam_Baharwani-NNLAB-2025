# Objective
WAP to evaluate the performance of implemented three-layer neural network with variations in activation functions, size of hidden layer, learning rate, batch size and number of epochs.

## Directory Structure
```
├── mnist_results/              # Output directory containing plots and final PDF report
│   ├── loss_<activation>.png    # Loss curve plot for each activation function
│   ├── conf_matrix_<activation>.png # Confusion matrix plot for each activation function
│   └── MNIST_Results_Report.pdf # Comprehensive report of all results
├── main.py                     # Python script for training and generating results
├── README.md                   # This README file
```

## Requirements
- Python 3.x
- TensorFlow (Compatible with 1.x style graph-based execution)
- NumPy
- Matplotlib
- scikit-learn
- seaborn
- fpdf

Install the dependencies via:
```bash
pip install tensorflow numpy matplotlib scikit-learn seaborn fpdf
```

## How to Run
1. Clone the repository:
```bash
git clone <repository-url>
```
2. Navigate to the project directory:
```bash
cd <project-directory>
```
3. Run the script:
```bash
python main.py
```

## Description
The model is a **Single Layer Neural Network** trained on the MNIST dataset with different activation functions:
- ReLU
- Sigmoid
- Tanh

### Model Architecture
- **Input Layer:** 784 neurons (Flattened 28x28 images)
- **Hidden Layer:** 256 neurons (Activation: ReLU/Sigmoid/Tanh)
- **Output Layer:** 10 neurons (Softmax activation for classification)
