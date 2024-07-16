import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Function to train the model (only for demonstration; ideally, you would load a pre-trained model)
def train_model():
    heart_data = pd.read_csv("heart.csv")  # Make sure you have the dataset in the same directory
    X = heart_data.drop(columns='target', axis=1)
    y = heart_data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

    clf = SVC(kernel="linear")
    clf.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(clf, 'heart_disease_model.pkl')
    
    y_pred = clf.predict(X_test)
    classification_rep = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(classification_rep)

# Train the model (uncomment this line if you need to train the model)
train_model()