
# NISHITA KADIYA
                         # codsoft
                     # TASK 1 : Titanic Servival Prediction

import pandas as pd  # for loading csv
from sklearn.model_selection import train_test_split # spliting traing and testing data
from sklearn.naive_bayes import GaussianNB # model that predict

import streamlit as st  # gives graphical interface

df = pd.read_csv("titanic.csv")  # reading csv file
print(df.head())

# Removing Unneccsary columns that are not useful for prediction
df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
print(df.head())

# making binary data for gender bcz it is not balanced so 0 for male and 1 for female
d = {'male':0,'female':1}
df['Gender'] = df['Gender'].map(d)
print(df.head())

target = df.Survived  # dependent variable
print(target)
 
input_var = df.drop('Survived',axis='columns') # independent variables

# handling missing value for column Age 
input_var.Age = input_var.Age.fillna(input_var.Age.mean())
print(input_var.head(10))

st.header(":blue[Titanic Servival Prediction]")

# split the csv dataset into training and testing part
x_train,x_test,y_train,y_test = train_test_split(input_var,target,test_size=0.2)

# intialize object of model

model = GaussianNB()

# load training data into model
model.fit(x_train,y_train)

# testing data
print(model.score(x_test,y_test))
print(y_test[:20])
print(model.predict(x_test)[:20])   # predicting xtest data based on ytest

# getting input from user
in_class = (st.number_input("Enter Passeneger Class : "))
gender = st.selectbox("Enter Gender : ",options=['male','female'])

if gender == 'male':
    in_gender = 0
else:
    in_gender = 1

in_age = (st.number_input('Enter Age  : '))
in_fare = (st.number_input('Enter Fare : '))
in_age = float(in_age)
in_fare = float(in_fare)

test = pd.DataFrame({'Pclass':[in_class],'Gender':[in_gender],'Age':[in_age],'Fare':[in_fare]})

a = model.predict(test)  # predicting testing data

if st.button("Make Prediction"):
    
    # if a==0 that meand not survived else servived
    if(a==0):
        st.subheader("Passenger will Not Survived")
    else:
        st.subheader('Passenger will Survived')