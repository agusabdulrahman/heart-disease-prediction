import numpy as np
import pandas as pd
import streamlit as st
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from PIL import Image

# Function to train the model (only for demonstration; ideally, you would load a pre-trained model)
# def train_model():
#     heart_data = pd.read_csv("heart.csv")  # Make sure you have the dataset in the same directory
#     X = heart_data.drop(columns='target', axis=1)
#     y = heart_data['target']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

#     clf = SVC(kernel="linear")
#     clf.fit(X_train, y_train)
    
#     # Save the model
#     joblib.dump(clf, 'heart_disease_model.pkl')
    
#     y_pred = clf.predict(X_test)
#     classification_rep = classification_report(y_test, y_pred)
#     print("Classification Report:")
#     print(classification_rep)

# Train the model (uncomment this line if you need to train the model)
# train_model()

# Load the model
model = joblib.load('model/heart_disease_model.pkl')

# Function to predict heart disease
def predict_heart_disease(model, input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    return prediction[0]

# Streamlit app

# Streamlit app
st.set_page_config(page_title="Heart Disease Prediction", layout="centered", initial_sidebar_state="expanded")

# Load image
image = Image.open("media/cover.jpg")
st.image(image, use_column_width=True)

st.title("Heart Disease Prediction")

# # Sidebar theme selectiongit 
# theme = st.sidebar.radio("Select Theme", ("Light", "Dark"))

# if theme == "Dark":
#     st.markdown("""
#         <style>
#         .reportview-container {
#             background: #2C2C2C;
#             color: white;
#         }
#         .sidebar .sidebar-content {
#             background: #1A1A1A;
#             color: white;
#         }
#         </style>
#         """, unsafe_allow_html=True)
# else:
#     st.markdown("""
#         <style>
#         .reportview-container {
#             background: white;
#             color: black;
#         }
#         .sidebar .sidebar-content {
#             background: #F0F0F5;
#             color: black;
#         }
#         </style>
#         """, unsafe_allow_html=True)
# # st.title("Heart Disease Prediction")

# Input fields
st.sidebar.header("Input Features")
age = st.sidebar.number_input('Age', min_value=0, max_value=120, value=25)
sex = st.sidebar.selectbox('Sex', [0, 1])
cp = st.sidebar.selectbox('Chest Pain Type', [0, 1, 2, 3])
trestbps = st.sidebar.number_input('Resting Blood Pressure', min_value=0, max_value=300, value=120)
chol = st.sidebar.number_input('Serum Cholestoral in mg/dl', min_value=0, max_value=600, value=200)
fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1])
restecg = st.sidebar.selectbox('Resting Electrocardiographic Results', [0, 1, 2])
thalach = st.sidebar.number_input('Maximum Heart Rate Achieved', min_value=0, max_value=250, value=150)
exang = st.sidebar.selectbox('Exercise Induced Angina', [0, 1])
oldpeak = st.sidebar.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, value=1.0)
slope = st.sidebar.selectbox('Slope of the Peak Exercise ST Segment', [0, 1, 2])
ca = st.sidebar.selectbox('Number of Major Vessels Colored by Flourosopy', [0, 1, 2, 3, 4])
thal = st.sidebar.selectbox('Thalassemia', [0, 1, 2, 3])

# Prediction button
if st.button('Predict'):
    input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    prediction = predict_heart_disease(model, input_data)
    
    if prediction == 0:
        st.success('The person does not have heart disease.')
    else:
        st.error('The person has heart disease.')

# Option to display the classification report
if st.checkbox('Show Classification Report'):
    heart_data = pd.read_csv("dataset/heart.csv")
    X = heart_data.drop(columns='target', axis=1)
    y = heart_data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

    y_pred = model.predict(X_test)
    classification_rep = classification_report(y_test, y_pred)
    st.text(classification_rep)
