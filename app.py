# app.py
# House Price Prediction Web App using Streamlit

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏡",
    layout="centered"
)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("🏡 House Price Prediction App")
st.write("Predict house prices using Machine Learning and Linear Regression.")

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

data = pd.read_csv("house_prices.csv")

# ---------------------------------------------------
# ENCODE LOCATION
# ---------------------------------------------------

encoder = LabelEncoder()
data['Location'] = encoder.fit_transform(data['Location'])

# ---------------------------------------------------
# FEATURES & TARGET
# ---------------------------------------------------

X = data[['Area', 'Location']]
y = data['Price']

# ---------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------

model = LinearRegression()
model.fit(X_train, y_train)

# ---------------------------------------------------
# USER INPUTS
# ---------------------------------------------------

st.header("Enter House Details")

area = st.number_input(
    "Enter Area (Square Feet)",
    min_value=500,
    max_value=10000,
    value=1900
)

location = st.selectbox(
    "Select Location",
    ['Karachi', 'Islamabad', 'Lahore']
)

# ---------------------------------------------------
# ENCODE USER LOCATION
# ---------------------------------------------------

location_encoded = encoder.transform([location])[0]

# ---------------------------------------------------
# PREDICT BUTTON
# ---------------------------------------------------

if st.button("Predict House Price"):

    # Create dataframe for prediction
    new_house = pd.DataFrame({
        'Area': [area],
        'Location': [location_encoded]
    })

    # Predict price
    predicted_price = model.predict(new_house)

    # Display result
    st.success(f"🏷 Predicted House Price: Rs. {round(predicted_price[0], 2)}")

  


# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")
st.write("Developed using Python, Streamlit & Machine Learning 🚀")
