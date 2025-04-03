# Experiment 3: Classifying Handwritten Digits with a Three-Layer Neural Network

## Objective

The goal of this experiment is to build a three-layer neural network from scratch using TensorFlow (without Keras!) to accurately classify handwritten digits from the MNIST dataset. We're tackling this problem by implementing the key steps of a neural network: feed-forward propagation, loss calculation, backpropagation, and optimization.

## Model Architecture

Here's the breakdown of our network:

- **Input Layer:** 784 neurons (flattened from 28 × 28 pixel images)
- **Hidden Layer 1:** 128 neurons, using ReLU activation for better performance
- **Hidden Layer 2:** 64 neurons, also using ReLU activation
- **Output Layer:** 10 neurons (one for each digit, 0–9)  
  *Note: No activation function here since we're handling Softmax through the loss function.*

## Hyperparameters

- **Epochs:** 20
- **Learning Rate:** 0.001 (with a fancy exponential decay scheduler to keep things optimal)
- **Batch Size:** 128
- **Loss Function:** Categorical Cross-Entropy
- **Optimizer:** Adam (for smoother, faster convergence)

## Improvements Over Previous Implementations

1. **ReLU Activation Functions:** Replacing Sigmoid to avoid vanishing gradients and boost performance.
2. **HeNormal Initialization:** Smart weight initialization for deeper networks.
3. **Batch Normalization:** Applied to each layer for faster training and better stability.
4. **Dropout Regularization:** Preventing overfitting by randomly dropping neurons during training.
5. **Learning Rate Scheduler:** Adjusting the learning rate dynamically for better performance.
6. **Adam Optimizer:** Faster training and improved convergence.

## Dependencies

- `TensorFlow`
- `NumPy`

### Installation

```bash
pip install tensorflow numpy
```

## Usage

The entire implementation is in the file `MNIST_Classification_Improved.py`. To run the experiment, just hit:

```bash
python MNIST_Classification_Improved.py
```

## Results

Thanks to the improvements we’ve made, the model should achieve impressive training and test accuracy. You’ll see detailed results printed out after training.

## Future Improvements

1. Trying deeper networks with additional hidden layers.
2. Experimenting with other optimizers like RMSprop or Nadam.
3. Incorporating convolutional layers for even better feature extraction.  
