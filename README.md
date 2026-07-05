# LSTM Sequence Forecasting: Air Astana Share Price

Forecasting Air Astana's daily share price with stacked LSTMs.

## What the notebook does

The notebook loads 526 daily closing prices for Air Astana covering February 2024 through March 2026, sorts them chronologically, and standardizes the series. It then frames forecasting as a sequence problem: sliding windows of past prices are used to predict the next day's close.

Two window lengths are compared, 12 days and 24 days. For each one, the notebook builds sequences, splits them chronologically into 70% training, 15% validation, and 15% test, and trains the same architecture: a stacked LSTM with 64 units feeding into 32 units and a single linear output. Training uses Adam with mean squared error loss for up to 20 epochs, with early stopping on validation loss to prevent overfitting. Each model is evaluated on its test segment with MAE and RMSE, and its predictions are plotted against the actual series.

The window with the lower RMSE is then retrained and saved as the final model.

## Results

The 12-day window won and its saved model reached an RMSE of 0.183 and an MAE of 0.141 on the test segment. Both numbers are in standardized units, since the series is scaled before training; they describe error relative to the spread of the price series rather than in tenge.

## Tech

Python, TensorFlow / Keras, scikit-learn, pandas

## Notes

The 12-day window won because with only about 500 observations, longer windows leave fewer training sequences and add noise instead of context. The biggest limitation is that errors are reported in standardized units instead of tenge, so they are hard to read as real money. Converting predictions back to prices and comparing against a naive baseline that repeats yesterday's price would show whether the model truly earns its keep. Models like this can be used in stock forecasting and demand planning.
