# Experiment 5: CNN on Fashion MNIST using Keras

## 🧪 Objective
To train and evaluate a Convolutional Neural Network (CNN) using the **Keras library** on the **Fashion MNIST** dataset. The experiment demonstrates the effect of varying:
- Filter size
- Regularization (L2)
- Batch size
- Optimization algorithm

## 🧠 Model Architecture
- **Conv2D Layer:** 32 filters, kernel size (3x3 or 5x5), ReLU activation
- **MaxPooling2D Layer:** 2x2 pooling
- **Flatten Layer**
- **Dense Layer:** 64 units, ReLU
- **Output Layer:** 10 units, Softmax

## 🛠️ Hyperparameters Tuned
- **Filter Sizes:** 3, 5
- **Regularization:** None, L2 (0.01)
- **Batch Sizes:** 32, 64
- **Optimizers:** Adam, SGD

## 📁 Outputs
- All accuracy plots are saved in the `exp5_outputs/` directory.
- File naming format:
  ```
  acc_f<filter>_r<reg>_b<batch_size>_o<optimizer>.png
  ```

## 🚀 How to Run
Install the required packages:
```bash
pip install tensorflow matplotlib
```
Then run the script:
```bash
python cnn_fashion_mnist_exp5.py
```

## 📊 Evaluation
Each model is trained for 5 epochs. Validation accuracy and training accuracy are plotted and saved for comparison.

## 📦 Dataset
- **Fashion MNIST:** 28x28 grayscale images of clothing items (10 classes).

## 📑 License
MIT License

