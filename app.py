import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
    

# PAGE SETUP
st.set_page_config(page_title="Diabetes App", page_icon="🩺", layout = "wide")


# Load the saved model
model = joblib.load(r'C:\Users\user\Documents\DATA ANALYTICS\MERISKILL\Project 2 - Diabetes Data\Logistic Regression.pkl')

# Load Dataset
data = pd.read_csv(r"C:\Users\user\Documents\DATA ANALYTICS\MERISKILL\Project 2 - Diabetes Data\diabetes.csv")

# Function to make predictions
def make_prediction(input_data):
    input_data = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_data)
    return prediction

# SIDEBAR NAVIGATION
with st.sidebar:
    st.markdown("## Menu")
    page = st.radio("Go to", ["Prediction", "Data", "Exploration"])
    st.markdown("---")
    st.markdown("#### Model Accuracy")
    st.markdown("76%")
    st.markdown("---")
    st.markdown("## Monday Ochedi")
    st.markdown(
    """
    <div style="margin-top:20px; text-align:left;">
        <ul style="list-style-type:none; padding:0; margin:0;">
            <li style="margin: 10px 0;">
                <a href="https://www.linkedin.com/in/monday-ochedi/" target="_blank" style="text-decoration:none;">
                    <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" style="vertical-align:middle; margin-right:10px;"/>
                    LinkedIn
                </a>
            </li>
            <li style="margin: 10px 0;">
                <a href="https://github.com/Monday-Ochedi" target="_blank" style="text-decoration:none;">
                    <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="25" style="vertical-align:middle; margin-right:10px;"/>
                    GitHub
                </a>
            </li>
            <li style="margin: 10px 0;">
                <a href="https://www.facebook.com/ochedi.monday.75" target="_blank" style="text-decoration:none;">
                    <img src="https://cdn-icons-png.flaticon.com/512/733/733547.png" width="25" style="vertical-align:middle; margin-right:10px;"/>
                    Facebook
                </a>
            </li>
            <li style="margin: 10px 0;">
                <a href="https://wa.me/+2348168794576" target="_blank" style="text-decoration:none;">
                    <img src="https://cdn-icons-png.flaticon.com/512/733/733585.png" width="25" style="vertical-align:middle; margin-right:10px;"/>
                    WhatsApp
                </a>
            </li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)



# PREDICTION PAGE
if page == "Prediction":
    st.markdown("# Diabetes Prediction System 🩺")
    
    st.write("Enter the values below to make a prediction:")

    # Define input fields for the model features
    Pregnancies = st.text_input("Pregnancies: Number of times you have been pregnant")
    Glucose = st.text_input("Glucose: Plasma glucose concentration after 2 hours in an oral glucose tolerance test")
    BloodPressure = st.text_input("BloodPressure: Diastolic blood pressure (mm Hg)")
    SkinThickness = st.text_input("SkinThickness: Triceps skinfold thickness (mm)")
    Insulin = st.text_input("Insulin: 2-hour serum insulin (mu U/ml)")
    BMI = st.text_input("BMI: Body Mass Index (weight in kg / (height in m)²)")
    DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction: Function Score of diabetes likelihood based on family history")
    Age = st.text_input("Age: How old are you (years)")

    # Create an input list
    user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]

    # Prediction button
    if st.button("**Make Prediction**"):
        if not all(user_input):
            st.warning("Please fill in all fields before making a prediction.")
        else:
            # Convert text inputs to numeric values
            user_input = [float(x) for x in user_input]

        # Make prediction
        prediction = make_prediction(user_input)
        
        # Display prediction result
        if prediction == 1:
            st.success("POSITIVE")
        else:
            st.success("NEGATIVE")



# DATA PAGE
elif page == "Data":
    st.markdown("# Data")

    with st.expander("Dataset"):
        data

        with st.expander("Predictor Variable (X)"):
            X = data.drop(['Outcome'], axis = 1)
            X

        with st.expander("Target Variable (y)"):
            y = data['Outcome']
            y
    with st.expander("Info"):
        st.write(data.dtypes)

    with st.expander("Summary Statistic"):
        st.write(data.describe())
        


#DATA EXPLORATION PAGE
else:
    st.markdown("# Data Exploration")

    with st.expander("Correlation Heatmap"):
        fig, ax= plt.subplots(figsize=(6, 2))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)

    with st.expander("Pregnancie vs Outcome"):
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.scatter(data = data, x = 'Pregnancies', y = 'Outcome')
        ax.set_xlabel("Pregnancies")
        ax.set_ylabel("Outcome")
        st.pyplot(fig)

    with st.expander("DiabetesPedigreeFunction vs Outcome"):
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.scatter(data = data, x = 'DiabetesPedigreeFunction', y = 'Outcome')
        ax.set_xlabel("DiabetesPedigreeFunction")
        ax.set_ylabel("Outcome")
        st.pyplot(fig)

    with st.expander("Age vs Outcome"):
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.scatter(data = data, x = 'Age', y = 'Outcome')
        ax.set_xlabel("Age")
        ax.set_ylabel("Outcome")
        st.pyplot(fig)