AI-Powered Coca-Cola Stock Prediction
Overview
This project utilizes machine learning to predict Coca-Cola's stock price based on historical data. The model uses a Linear Regression approach to forecast future stock prices. The data is fetched from yfinance for the Coca-Cola stock (KO) listed on the NYSE.

Steps
Fetch Data: The data for Coca-Cola's stock is fetched using the yfinance library.

Preprocess Data: Clean the data by removing any missing values and converting the date into a numerical format suitable for machine learning models.

Train the Model: A Linear Regression model is used to train on historical stock data and predict future stock prices.

Prediction: The model forecasts stock prices for the next 5 days.

Visualization: The actual vs. predicted stock prices are visualized to evaluate the performance of the model.

Requirements
Before running the project, ensure you have the following Python libraries installed:

yfinance (for fetching stock data)

pandas (for data manipulation)

scikit-learn (for machine learning model)

matplotlib (for data visualization)

You can install them using:

bash
Copy
Edit
pip install yfinance pandas scikit-learn matplotlib
Code Walkthrough
Step 1: Fetch Coca-Cola's Stock Data
python
Copy
Edit
import yfinance as yf

# Fetching Coca-Cola stock data (KO) from the NYSE
coca_cola_stock = yf.Ticker("KO")

# Getting historical market data for Coca-Cola stock (last 1 year)
coca_cola_data = coca_cola_stock.history(period="1y")

# Displaying the first few rows of the data
print(coca_cola_data.head())
Step 2: Clean & Preprocess the Data
python
Copy
Edit
# Check for missing values
print(coca_cola_data.isnull().sum())

# Drop rows with missing values
coca_cola_data = coca_cola_data.dropna()

# Use only 'Close' price for prediction
coca_cola_data = coca_cola_data[['Close']]

# Reset the index and convert date to a numerical format
coca_cola_data = coca_cola_data.reset_index()
coca_cola_data['Date'] = coca_cola_data['Date'].map(lambda x: x.timestamp())
Step 3: Train a Machine Learning Model
python
Copy
Edit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Features and target variable
X = coca_cola_data[['Date']]
y = coca_cola_data['Close']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
Step 4: Predict Future Stock Prices
python
Copy
Edit
import numpy as np

# Predict future stock prices (next 5 days)
future_dates = np.array([coca_cola_data['Date'].max() + (i * 86400) for i in range(1, 6)]).reshape(-1, 1)

# Make predictions
future_predictions = model.predict(future_dates)

print(f"Predicted future prices: {future_predictions}")
Step 5: Visualize the Results
python
Copy
Edit
import matplotlib.pyplot as plt

# Plot actual vs predicted prices
plt.figure(figsize=(10,6))
plt.plot(coca_cola_data['Date'], y, label="Actual Prices", color="blue")
plt.plot(X_test, y_pred, label="Predicted Prices", color="red")
plt.title("Coca-Cola Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.show()
Conclusion
This project successfully predicts Coca-Cola's stock price using historical data and machine learning techniques. The model can be expanded with more complex models (e.g., XGBoost, LSTM) for improved accuracy. Further improvements can include data from multiple sources or adding more features like market sentiment, global events, or competitor stock data.

Next Steps
Explore additional models such as XGBoost or LSTM for time series forecasting.

Integrate real-time stock data for live predictions.

Implement model optimization techniques to improve performance.
