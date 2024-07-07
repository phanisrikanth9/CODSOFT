# app.py
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained XGBoost model
model = joblib.load('xgboost_model.pkl')

# Function to preprocess data
def preprocess_data(data):
    # Drop unnecessary columns
    data = data.drop(['nameOrig', 'nameDest'], axis=1)
    
    # Encode categorical variables
    le = LabelEncoder()
    data['type'] = le.fit_transform(data['type'])  # Assuming 'type' is categorical
    
    return data

# Function to display required attributes (column names)
def display_required_attributes():
    required_columns = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                        'oldbalanceDest', 'newbalanceDest', 'isFraud', 'isFlaggedFraud']
    
    st.subheader('Required Attributes (Column Names)')
    st.write(required_columns)

# Function to create and display a pie chart
def create_pie_chart(predictions):
    fraud_count = sum(predictions)
    non_fraud_count = len(predictions) - fraud_count
    labels = ['Fraudulent', 'Non-Fraudulent']
    sizes = [fraud_count, non_fraud_count]
    colors = ['red', 'green']
    explode = (0.1, 0)  # explode the first slice (Fraudulent)
    
    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct='%1.1f%%', shadow=True, startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    st.pyplot(fig)

# Function to create and display a confusion matrix
def create_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

# Main function to run the Streamlit app
def main():
    st.title('Online Transaction Fraud Detection')
    
    # Display required attributes (column names)
    display_required_attributes()

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        # Preprocess the uploaded data
        data_preprocessed = preprocess_data(data)
        
        # Ensure 'isFraud' column is included
        if 'isFraud' not in data_preprocessed.columns:
            st.error("Dataset must include 'isFraud' column.")
            return
        
        # Display dataset information
        st.subheader('Uploaded Dataset')
        st.write(data_preprocessed.head())  # Display the first few rows of the dataset
        
        # Button to detect fraud
        if st.button('Detect Fraud'):
            try:
                # Perform predictions using the model
                X = data_preprocessed.drop(['isFraud'], axis=1)
                y_true = data_preprocessed['isFraud']
                predictions = model.predict(X)
                
                # Display results
                st.subheader('Fraud Detection Results')
                
                # Calculate and display fraud percentage
                fraud_percentage = predictions.mean() * 100
                st.write(f"Percentage of fraudulent transactions: {fraud_percentage:.2f}%")
                
                # Display number of fraud transactions detected
                st.write("Number of fraud transactions detected:", sum(predictions))
                
                # Display number of non-fraud transactions detected
                st.write("Number of non-fraud transactions detected:", len(predictions) - sum(predictions))
                
                # Show fraudulent transactions
                st.subheader('Fraudulent Transactions')
                fraud_indices = data_preprocessed[data_preprocessed['isFraud'] == 1].index
                fraudulent_transactions = data_preprocessed.loc[fraud_indices]
                st.write(fraudulent_transactions[['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                                                  'oldbalanceDest', 'newbalanceDest']])
                
                # Create and display a pie chart
                st.subheader('Fraudulent vs Non-Fraudulent Transactions')
                create_pie_chart(predictions)
                
                # Create and display confusion matrix
                st.subheader('Confusion Matrix')
                create_confusion_matrix(y_true, predictions)
                
                # Calculate and display F1 score
                f1 = f1_score(y_true, predictions)
                st.write(f"F1 Score: {f1:.2f}")
                
            except Exception as e:
                st.error(f"Error processing data: {e}")

# Run the app
if __name__ == "__main__":
    main()
