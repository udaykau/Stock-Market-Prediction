# Stock Price Prediction: LSTM and ARIMA Models

This repository contains two Jupyter notebooks for stock price prediction using two different approaches: an LSTM (Long Short-Term Memory) neural network and an ARIMA (AutoRegressive Integrated Moving Average) statistical model. Both notebooks use historical stock data from Yahoo Finance, focusing on Apple Inc. (AAPL) and other major tech stocks.

---

## Contents

- [`LSTM-model.ipynb`](LSTM.ipynb): Deep learning approach using LSTM for time series forecasting.
- [`ARIMA-model.ipynb`](ARIMA.ipynb): Statistical approach using ARIMA for time series forecasting.

---

## LSTM Model (`LSTM.ipynb`)

### Overview

- Fetches historical daily stock data for AAPL, GOOG, MSFT, and AMZN.
- Visualizes closing prices, trading volumes, moving averages, and daily returns.
- Prepares the data for LSTM by scaling and creating sequences.
- Builds and trains an LSTM neural network to predict the closing price of AAPL.
- Evaluates the model using RMSE and MAPE.
- Visualizes actual vs. predicted prices.

### Key Steps

1. **Data Collection**: Downloads data from Yahoo Finance using pandas.
2. **Visualization**: Plots closing prices, volumes, moving averages, and return distributions.
3. **Feature Engineering**: Calculates moving averages and daily returns.
4. **Data Preparation**: Scales data and creates training/testing sequences.
5. **Model Building**: Constructs an LSTM model using Keras.
6. **Training**: Trains the model and plots the loss curve.
7. **Evaluation**: Computes RMSE and MAPE, and visualizes predictions.

### Requirements

- Python 3.x
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- keras, tensorflow

---

## ARIMA Model (`ARIMA.ipynb`)

### Overview

- Fetches historical daily stock data for AAPL.
- Visualizes closing prices and their distribution.
- Tests for stationarity and decomposes the time series.
- Applies log transformation and differencing as needed.
- Fits an ARIMA model (with auto_arima for parameter selection).
- Forecasts future prices and evaluates the model using MSE, MAE, RMSE, and MAPE.
- Compares actual vs. forecasted values.

### Key Steps

1. **Data Collection**: Downloads AAPL data from Yahoo Finance.
2. **Visualization**: Plots closing prices and their KDE distribution.
3. **Stationarity Testing**: Uses rolling statistics and Dickey-Fuller test.
4. **Decomposition**: Decomposes the series into trend, seasonality, and residuals.
5. **Transformation**: Applies log transformation and differencing.
6. **Model Selection**: Uses auto_arima to select optimal parameters.
7. **Model Fitting**: Fits ARIMA and forecasts future values.
8. **Evaluation**: Computes error metrics and displays comparison DataFrame.

### Requirements

- Python 3.x
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- statsmodels
- pmdarima

---

## Usage

1. Clone the repository or download the notebooks.
2. Install the required Python packages (see above).
3. Open the notebooks in Jupyter or VS Code.
4. Run the cells sequentially to reproduce the analysis and results.

---

## Notes

- Both notebooks fetch data directly from Yahoo Finance; ensure you have an internet connection.
- The LSTM notebook focuses on deep learning for AAPL but can be adapted for other stocks.
- The ARIMA notebook demonstrates a classical time series approach for AAPL.

---

## License

This project is for academic and educational purposes.
