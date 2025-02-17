# Multi-Layer Perceptron (MLP) for XOR using Step Function

## Objective
To implement a **Multi-Layer Perceptron (MLP) network with one hidden layer** using **NumPy** in Python. The model is trained to learn the **XOR Boolean function** using a **step activation function**.

## Description
The **XOR function** is a classic problem that cannot be solved by a single-layer perceptron. A **multi-layer perceptron (MLP)** with a **hidden layer** is required to learn the non-linearity of the XOR function.

### Key Features:
- **Step Activation Function**: Instead of sigmoid or ReLU, a step function is used to classify outputs.
- **Custom Weight Updates**: Since the step function is non-differentiable, a perceptron-like weight update rule is used.
- **One Hidden Layer**: Allows the network to capture XOR’s non-linearity.
- **Early Stopping**: Training stops when error reaches zero.

## Implementation Details
- **Input Layer**: Two neurons (for two input features)
- **Hidden Layer**: Two neurons (to capture non-linearity)
- **Output Layer**: One neuron (final XOR classification)
- **Learning Rate**: 0.1
- **Epochs**: 10,000 (or stops early if error reaches zero)

## How the Algorithm Works
1. **Initialize Weights & Biases** randomly.
2. **Forward Pass:**
   - Compute hidden layer input and apply the **step activation function**.
   - Compute output layer input and apply the **step activation function**.
3. **Error Calculation:**
   - Compute the difference between the predicted and actual output.
4. **Weight Update:**
   - Adjust weights and biases using a modified perceptron learning rule.
5. **Repeat until convergence** (or max epochs reached).

## XOR Dataset
| Input 1 | Input 2 | Output |
|---------|---------|--------|
| 0       | 0       | 0      |
| 0       | 1       | 1      |
| 1       | 0       | 1      |
| 1       | 1       | 0      |

## Code
The implementation is available in `mlp_xor.py`. Run the script to train and test the MLP on the XOR dataset.

```bash
python mlp_xor.py
```

## Expected Output
```plaintext
Epoch 0: Total Error = 3
Epoch 1000: Total Error = 2
Epoch 2000: Total Error = 1
Epoch 3000: Total Error = 0

Testing the trained MLP on XOR dataset:
Input: [0 0], Predicted Output: 0
Input: [0 1], Predicted Output: 1
Input: [1 0], Predicted Output: 1
Input: [1 1], Predicted Output: 0
```

## Conclusion
This experiment demonstrates that an MLP with a **hidden layer** and **step activation function** can successfully learn the **XOR Boolean function**. The training process updates weights iteratively, and the network converges to the correct output after a certain number of epochs.

---
### Author: Shivam Baharwani
### Technologies Used: Python, NumPy

