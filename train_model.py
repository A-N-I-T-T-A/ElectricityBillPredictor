import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("electricity_bill.csv")

# Assuming dataset has 'units' and 'bill_amount' columns
X = df[['Units_Consumed']]
y = df['Electricity_Bill']

# Train a simple Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Save the trained model
joblib.dump(model, "model.pkl")
print("Model training completed and saved as model.pkl")
