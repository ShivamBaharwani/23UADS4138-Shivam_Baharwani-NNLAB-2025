# Perceptron Learning Algorithm

## Objective
To implement the **Perceptron Learning Algorithm** using NumPy in Python and evaluate its performance on **NAND** and **XOR** truth tables as input datasets.

## Description
A **Perceptron** is a fundamental unit of a neural network used for binary classification. It applies a linear decision boundary to separate classes. This experiment demonstrates how a single-layer perceptron learns to classify data using the perceptron learning rule.

## Implementation Details
- Implemented a **Perceptron class** with weight initialization, activation function, prediction, and training methods.
- Used **NumPy** for efficient numerical operations.
- Evaluated the perceptron on two logic functions:
  - **NAND Gate** (Linearly Separable)
  - **XOR Gate** (Not Linearly Separable)

## Dataset
The input consists of two binary features (0 or 1), and the expected outputs follow the respective truth tables:

### NAND Gate Truth Table
| Input 1 | Input 2 | Output |
|---------|---------|--------|
| 0       | 0       | 1      |
| 0       | 1       | 1      |
| 1       | 0       | 1      |
| 1       | 1       | 0      |

### XOR Gate Truth Table
| Input 1 | Input 2 | Output |
|---------|---------|--------|
| 0       | 0       | 0      |
| 0       | 1       | 1      |
| 1       | 0       | 1      |
| 1       | 1       | 0      |

## Execution
1. **Run the script**: The perceptron is trained using the NAND and XOR datasets.
2. **Observe the performance**:
   - The perceptron successfully learns the **NAND** function.
   - The perceptron fails to learn **XOR**, as it is not linearly separable.

## Results
- The perceptron correctly classifies **NAND** gate outputs.
- The perceptron fails to classify **XOR** due to its non-linearity.
- This highlights the **limitation of a single-layer perceptron**, emphasizing the need for multi-layer networks for complex decision boundaries.

## Conclusion
The experiment demonstrates how a perceptron learns linearly separable functions but struggles with non-linearly separable ones like XOR. To solve such problems, **multi-layer perceptrons (MLPs) with activation functions like ReLU or sigmoid** are required.

## Requirements
- Python 3.x
- NumPy

## How to Run
```sh
python perceptron.py
```

## Future Enhancements
- Implement a multi-layer perceptron (MLP) to solve XOR.
- Use different activation functions like sigmoid or ReLU.
- Evaluate performance with more complex datasets.

---
**Author:** Shivam Baharwani  
**Date:** February 2025

