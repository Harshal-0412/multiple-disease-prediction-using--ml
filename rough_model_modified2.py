# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 19:23:12 2022

@author: DELL
"""

import pickle
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# loading the saved models
#loading the saved model
diabetes_loaded_model = pickle.load(open('D:/multiple disease prediction project/trained_model4.sav', 'rb'))

heart_loaded_model = pickle.load(open('D:/multiple disease prediction project/heart_savfile.sav', 'rb'))

breast_loaded_model = pickle.load(open('D:/multiple disease prediction project/breat_cancer_random_forest_algorithm.sav', 'rb'))

#creating the function for prediction

def diabetes_prediction(input_data):
    
    #input_data = (5,166,72,19,175,25.8,0.587,51)
    #input_data = (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)

    # changing the input to the numpy array
    input_data_as_numpy_array1 = np.asarray(input_data)

    # reshape the arrayas we are predicting for one instance
    input_data_reshaped1 = input_data_as_numpy_array1.reshape(1, -1)


    prediction = diabetes_loaded_model.predict(input_data_reshaped1)
    #print(prediction)

    if prediction[0] == 0:
        return "the person is non diabetic"
    else:
        return "the person is diabetic"
    
def heart_prediction(input_data):
    
    #input_data = (71,0,0,112,149,0,1,125,0,1.6,1,0,2)

    #changing the input data to the numpy data
    input_data_as_numpy_array2 = np.asarray(input_data)

    #reshape the array for one instance
    input_data_reshaped2 = input_data_as_numpy_array2.reshape(1, -1)

    prediction = heart_loaded_model.predict(input_data_reshaped2)
    print(prediction)

    if prediction[0] == 0:
        return "heart disease : **negative**"
    else:
        return "heart disease : **positive**"

def breast_prediction(input_data):
    
    #input_data = (71,0,0,112,149,0,1,125,0,1.6,1,0,2)

    #changing the input data to the numpy data
    input_data_as_numpy_array3 = np.asarray(input_data)

    #reshape the array for one instance
    input_data_reshaped3 = input_data_as_numpy_array3.reshape(1, -1)
    
    prediction = breast_loaded_model.predict(input_data_reshaped3)
    print(prediction)

    if prediction[0] == 0:
        return "Breast cancer : **Negative**"
    else:
        return "Breast cancer : **Positive**"


def main():
    
    # sidebar for navigation
    with st.sidebar:
        selected = option_menu('Multiple Disease Prediction System',

                               ['Diabetes Prediction',
                                'Heart Disease Prediction',
                                'Breast Cancer Prediction',],
                               icons=['activity', 'heart', 'circle'],
                               default_index=0)
    
    
    # Diabetes Prediction Page
    if selected == 'Diabetes Prediction':
        
        #giving a title
        st.title('diabetes prediction web app')
    
        # getting the input data from the user
        col1, col2, col3 = st.columns(3)
        #getting the input data from the user
        
        with col1:
            Pregnancies = st.text_input('number of pregnancies :: ')
            
        with col2:
            Glucose = st.text_input('Glucose value :: ')
    
        with col3:
            BloodPressure = st.text_input('BloodPressure range :: ')
    
        with col1:
            SkinThickness = st.text_input('SkinThickness :: ')
    
        with col2:
            Insulin = st.text_input('Insulin level :: ')
            
        with col3:
            BMI = st.text_input('BMI index :: ')
    
        with col1:
            DiabetesPedigreeFunction = st.text_input('DiabetesPedigreeFunction index :: ')
    
        with col2:
            Age = st.text_input('age of person :: ')
    
        # code for prediction
        diagnosis = ''
        
        # creating a button for prediction
        if st.button("diabetes test result"):
            input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
            diagnosis = diabetes_prediction(input_data)
        
        st.success(diagnosis)
    
    
    if selected == 'Breast Cancer Prediction':

        # page title
        st.title('Breast Cancer Prediction using ML')
        
        col1, col2, col3 = st.columns(3)
           
        with col1:
            #Clump_Thickness = st.text_input('Clunp thickness')
            Clump_Thickness = st.slider('Clunp thickness', 0, 15)
            #st.write(Clump_Thickness)
            
        with col2:
            #Cell_Size = st.text_input('Cell Size')
            Cell_Size = st.slider('Cell Size', 0, 15)
        
        with col3:
            #Cell_Shape = st.text_input('Cell Shape')
            Cell_Shape = st.slider('Cell Shape', 0, 15)
        
        with col1:
            #Marginal_Adhesion = st.text_input('Marginal Adhesion')
            Marginal_Adhesion = st.slider('Marginal Adhesion', 0, 15)
        
        with col2:
            #Single_Epithelial_Cell_Size = st.text_input('Single Epithelial Cell Size')
            Single_Epithelial_Cell_Size = st.slider('Single Epithelial Cell Size', 0, 15)
        
        with col3:
            #Bare_Nuclei = st.text_input('Bare Nuclei')
            Bare_Nuclei = st.slider('Bare Nuclei', 0, 15)
        
        with col1:
            #Bland_Chromatin = st.text_input('Bland Chromatin')
            Bland_Chromatin = st.slider('Bland Chromatin', 0, 15)
        
        with col2:
            #Normal_Nucleoli = st.text_input('Normal Nucleoli')
            Normal_Nucleoli = st.slider('Normal Nucleoli', 0, 15)
        
        with col3:
            #Mitoses = st.text_input('Mitoses')
            Mitoses = st.slider('Mitoses', 0, 15)
        
        diagnosis = ''
        
        # creating a button for prediction
        if st.button("Breast cancer test result"):
            input_data = [Clump_Thickness, Cell_Size, Cell_Shape, Marginal_Adhesion, Single_Epithelial_Cell_Size, Bare_Nuclei, Bland_Chromatin, Normal_Nucleoli, Mitoses]
            diagnosis = breast_prediction(input_data)
        
        st.success(diagnosis)

    if selected == 'Heart Disease Prediction':
        
        # page title
        # Heart Disease Prediction Page
        st.title('Heart Disease Prediction using ML')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.text_input('Age')

        with col2:
            sexy = st.selectbox('Sex', ('0) female', '1) male'))

        with col3:
            cpy = st.selectbox('Chest Pain types', ('0) typical angina', '1) atypical angina', '2) non-anginal pain', '3) asymptomatic'))
        
        with col1:
            trestbps = st.text_input('Resting Blood Pressure')

        with col2:
            chol = st.text_input('Serum Cholestoral in mg/dl')

        with col3:
            fbsy = st.selectbox('Fasting Blood Sugar > 120 mg/dl',('0) no', '1) yes'))

        with col1:
            restecgy = st.selectbox('Resting Electrocardiographic results', ('0) normal', '1) having ST-T wave abnormality', '2) showing probable or definite left ventricular hypertrophy by Estes criteria'))

        with col2:
            thalach = st.text_input('Maximum Heart Rate achieved')

        with col3:
            exangy = st.selectbox('Exercise Induced Angina', ('0) no', '1) yes'))
            
        with col1:
            oldpeak = st.text_input('ST depression induced by exercise')

        with col2:
            slopey = st.selectbox('Slope of the peak exercise ST segment', ('0) upsloping', '1) flat', '2) downsloping'))
        
        with col3:
            ca = st.text_input('Major vessels colored by flourosopy')

        with col1:
            #thal = st.text_input('thal')
            thaly = st.selectbox('thal', ('01) normal', '2) fixed defect', '3) reversable defect'))
            
        if sexy == '0) female':
            sex = 0
        else:
            sex = 1
        
        if cpy == '0) typical angina':
            cp = 0
        elif cpy == '1) atypical angina':
            cp = 1
        elif cpy == '2) non-anginal pain':
            cp = 2
        else:
            cp = 3
            
        if fbsy == '0) no':
            fbs = 0
        else:
            fbs = 1
            
        if restecgy == '0) normal':
            restecg = 0
        elif restecgy == '1) having ST-T wave abnormality':
            restecg = 1
        else:
            restecg = 2
        
        if exangy == '0) no':
            exang = 0
        else:
            exang = 1
            
        if slopey == '0) upsloping':
            slope = 0
        elif slopey == '1) flat':
            slope = 1
        else:
            slope = 2
            
        if thaly == '01) normal':
            thal = 1
        elif thaly == '2) fixed defect':
            thal = 2
        else:
            thal = 3


        diagnosis = ''
            
        if st.button('heart test result'):
            input_data = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
            diagnosis = heart_prediction(input_data)
            
        st.success(diagnosis)
        

if __name__ == '__main__':
    main()

    

        
