import pandas as pd
import streamlit as st 
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


loan = pd.read_csv("train.csv")

 
def main():
      # giving the webpage a title
      
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="background-color:purple;padding:13px">
    <h1 style ="color:white;text-align:center;">Loan Status Prediction </h1>
    </div>
    """
      
    # this line allows us to display the front end aspects we have 
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html = True)
if __name__=='__main__':
	main()

	
Gender = st.selectbox('Gender',("Male","Female"))
if Gender == 'Male':
	Gender = 0
else :
	Gender = 1
MaritalStatus = st.selectbox('MaritalStatus',("Unmarried","Married"))
if MaritalStatus == 'Unmarried':
	MaritalStatus = 0
else :
	MaritalStatus = 1
Education = st.selectbox('Education',("Not Graduate","Graduate"))
if Education == 'Not Graduate':
	Education = 0
else :
	Education = 1
Self_Employed = st.selectbox('Self_Employed',("Yes","No"))
if Self_Employed == 'No':
	Self_Employed = 0
else :
	Self_Employed = 1
Dependents = st.selectbox('Dependents',('0','1','2'))
ApplicantIncome = st.number_input("Applicant Income")
CoApplicantIncome = st.number_input("Co-Applicant Income")
LoanAmount = st.number_input("Loan Amount")
	
	
	
from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
loan['Education']=label_encoder.fit_transform(loan['Education'])
loan['Gender']=label_encoder.fit_transform(loan['Gender'])
loan['Married']=label_encoder.fit_transform(loan['Married'])
loan['Self_Employed']=label_encoder.fit_transform(loan['Self_Employed'])


loan.drop(["Loan_ID"],inplace=True,axis = 1)
from sklearn import preprocessing
mode=loan['Gender'].mode() #Filling null values of categorical values columns with mode & continuos values column with mean values of that column
loan['Gender']=loan['Gender'].fillna(mode.iloc[0])
mode1=loan['Married'].mode()
loan['Married']=loan['Married'].fillna(mode1.iloc[0])
mode2=loan['Self_Employed'].mode()
loan['Self_Employed']=loan['Self_Employed'].fillna(mode2.iloc[0])
mean=loan['LoanAmount'].mean()
loan['LoanAmount']=loan['LoanAmount'].fillna(mean)
loan['Dependents'] =loan['Dependents'].replace({'3+': 0})
loan = loan.dropna()




	

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
array = loan.values
X = array[:,0:8]
Y = array[:,11]
seed = 7


kfold = KFold(n_splits=10, random_state=seed,shuffle=True)
cart = DecisionTreeClassifier()
num_trees = 100
model= BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
model.fit(X,Y)

inputs = [[Gender, MaritalStatus, Education,Self_Employed,Dependents,ApplicantIncome,CoApplicantIncome, LoanAmount]]
if st.button('Predict'):
	result = model.predict(inputs)
	updated_res = result.astype(str)
	st.success('Loan Status is {}'.format(updated_res))

