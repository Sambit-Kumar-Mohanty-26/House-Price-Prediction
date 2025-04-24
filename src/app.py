import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.title("House Price Prediction")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    required_columns = ['GrLivArea', 'SalePrice']
    if not all(col in data.columns for col in required_columns):
        st.error(f"Missing required columns: {required_columns}")
    else:
        data = data[required_columns].dropna()

        st.write("Dataset Preview:")
        st.dataframe(data.head())

    
        st.subheader("Living Area vs Sale Price")
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='GrLivArea', y='SalePrice', data=data)
        plt.title("Living Area vs Sale Price")
        plt.xlabel("Living Area (GrLivArea)")
        plt.ylabel("Sale Price")
        st.pyplot(plt)

        
        X = data[['GrLivArea']]
        y = data['SalePrice']

    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        st.write(f"R² Score: {r2:.4f}")
        st.write(f"MSE: {mse:.4f}")

    
        st.subheader("Make a Prediction")
        living_area = st.number_input("Enter Living Area (GrLivArea):", min_value=0.0, step=1.0)
        if st.button("Predict"):
            prediction = model.predict([[living_area]])
            st.write(f"Predicted Sale Price: ${prediction[0]:,.2f}")