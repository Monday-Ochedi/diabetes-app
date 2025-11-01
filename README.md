# 🩺 Diabetes Prediction System  

This Diabetes Prediction App is a machine learning web application built with Streamlit that predicts the likelihood of diabetes based on user input parameters such as glucose level, BMI, age, and other clinical features. The prediction model is trained using the Pima Indians Diabetes Dataset, a well-known dataset from the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK).

## App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://diabetes-app.streamlit.app/)

## What Is Diabetes?
Diabetes Mellitus is a chronic medical condition that occurs when the body cannot properly process blood glucose (sugar). It happens due to either insufficient insulin production (Type 1 diabetes) or the body’s inability to use insulin effectively (Type 2 diabetes). If left unmanaged, diabetes can lead to severe complications such as heart disease, kidney failure, and nerve damage.

## Importance of Early Detection
Early detection of diabetes is essential because:
- It allows for timely medical intervention and lifestyle changes.
- It helps reduce the risk of long-term complications.
- It supports better health management and monitoring of glucose levels.

**This project aims to provide an AI-powered tool that assists in early identification of potential diabetes risk using patient health data.** 

## How It Works
1. Users input clinical data such as:
  - Number of pregnancies
  - Glucose concentration
  - Blood pressure
  - Skin thickness
  - Insulin level
  - Body Mass Index (BMI)
  - Diabetes pedigree function
  - Age
2. The trained Logistic Regression model processes these inputs.
3. The model outputs a prediction indicating whether the person is likely diabetic or non-diabetic.


## Technologies Used
1. Python – Data preprocessing, model training, and backend logic
2. Scikit-learn – Machine learning (Logistic Regression model)
3. Streamlit – Web application interface
4. Pandas, NumPy – Data manipulation and numerical computations
5. Matplotlib, Seaborn – Data visualization


## Dataset
The dataset used for training is the Pima Indians Diabetes Dataset, which contains diagnostic measurements of women aged 21 years and older of Pima Indian heritage.
- Source: [here](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

## ⚠️Disclaimer
This app is intended for educational and research purposes only.
It is not a substitute for professional medical advice or diagnosis. Always consult a certified healthcare provider for any health concerns.
