# Experiment 6: Time Series Forecasting with RNN

## 🧪 Objective
Train and evaluate a **Recurrent Neural Network (RNN)** using Keras to predict the **next value** in a sample **time series dataset**.

## 🔧 Description
This experiment uses a synthetic sine-based signal to demonstrate sequence prediction using RNNs.

### Data
- **Synthetic sine wave** with two frequencies.
- Normalized using `MinMaxScaler`.
- Converted into supervised format using a **sliding window**.

### Model Architecture
- **Input:** Window of 20 time steps
- **RNN Layer:** 50 hidden units, tanh activation
- **Output Layer:** Dense(1) predicting the next value
- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Adam

## 📁 Output Files (saved in `exp6_outputs/`):
- `prediction_vs_actual.png` → Line plot comparing true vs predicted values.
- `loss_plot.png` → Training and validation loss over epochs.

## 🚀 How to Run
Install dependencies:
```bash
pip install tensorflow numpy matplotlib scikit-learn
```
Run the script:
```bash
python rnn_timeseries_exp6.py
```

## 📊 Evaluation
The model's accuracy is visually evaluated by comparing predicted and true values on a held-out test set.

## 📑 License
MIT License

