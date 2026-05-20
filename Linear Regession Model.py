# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset from CSV file
data = pd.read_csv("house_prices.csv")

print("Dataset:")
print(data)

# Convert location names into numbers
# Example:
# Karachi = 0
# Islamabad = 1
# Lahore = 2

encoder = LabelEncoder()
data['Location'] = encoder.fit_transform(data['Location'])

# X = Input Features
# y = Output/Target

# Features:
# Area + Location

X = data[['Area', 'Location']]

# Target:
# House Price

y = data['Price']

# Split data into training and testing
# 80% training
# 20% testing

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train model using training data
model.fit(X_train, y_train)

# Predict prices for test data
y_pred = model.predict(X_test)

# Show model performance
print("\nModel Performance:")
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# ---------------------------------------------
# LINEAR REGRESSION FORMULA
# ---------------------------------------------

# Formula:
# y = m1x1 + m2x2 + c

# Where:
# y  = Predicted House Price
# x1 = Area
# x2 = Location
# m1 = Coefficient of Area
# m2 = Coefficient of Location
# c  = Intercept

# ---------------------------------------------
# Predict price for a new house
# ---------------------------------------------
# User input:
# Area = 1900
# Location = Lahore
# Convert Lahore into encoded number

new_location = encoder.transform(['Lahore'])[0]

# Create new house data

new_house = pd.DataFrame({
    'Area': [1900],
    'Location': [new_location]
})

# Predict house price

predicted_price = model.predict(new_house)

# Show prediction

print("\nPredicted House Price:")
print("Rs.", round(predicted_price[0], 2))

