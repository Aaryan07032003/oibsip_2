import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import streamlit as st

file_path = 'Unemployment in India.csv'  
unemployment_data = pd.read_csv(file_path)
unemployment_data[' Date'] = pd.to_datetime(unemployment_data[' Date'])

st.title("UNEMPLOYMENT ANALYSIS DASHBOARD")

st.header("Basic Information")
st.write(unemployment_data.head())
st.write(unemployment_data.info())
st.write(unemployment_data.describe())

st.header("Missing Values")
st.write(unemployment_data.isnull().sum())

# Histogram
st.header("Distribution of Unemployment Rate")
plt.figure(figsize=(8, 6))
sns.histplot(data=unemployment_data, x=' Estimated Unemployment Rate (%)', bins=20)
plt.title('Histogram of Unemployment Rate (%) vs. Frequency')
plt.xlabel('Estimated Unemployment Rate (%)')
plt.ylabel('Frequency')
st.pyplot()

# Preprocessing
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(unemployment_data[[' Estimated Employed', ' Estimated Labour Participation Rate (%)']])
y_imputed = imputer.fit_transform(unemployment_data[' Estimated Unemployment Rate (%)'].values.reshape(-1, 1)).ravel()

model = LinearRegression()

y_pred = cross_val_predict(model, X_imputed, y_imputed, cv=5) 
mse = mean_squared_error(y_imputed, y_pred)
r2 = r2_score(y_imputed, y_pred)

st.header("Model Evaluation")
st.write("Mean Squared Error:", mse)
st.write("R-squared:", r2)

st.header("Visualization")
plt.figure(figsize=(10, 6))
plt.plot(unemployment_data[' Date'], unemployment_data[' Estimated Unemployment Rate (%)'], marker='o', linestyle='-')
plt.title('Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Estimated Unemployment Rate (%)')
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot()


