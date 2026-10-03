import os
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
from streamlit_option_menu import option_menu
import pickle
from PIL import Image
import numpy as np
import plotly.figure_factory as ff
import seaborn as sns
import joblib
from code.DiseaseModel import DiseaseModel
from code.helper import prepare_symptoms_array

# --- SETUP ABSOLUTE PATHS FOR STREAMLIT CLOUD ---
# This ensures files are found whether running locally or on Streamlit Cloud
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# loading the models
diabetes_model = joblib.load(os.path.join(BASE_DIR, "models", "diabetes_model.sav"))
heart_model = joblib.load(os.path.join(BASE_DIR, "models", "heart_disease_model.sav"))
parkinson_model = joblib.load(os.path.join(BASE_DIR, "models", "parkinsons_model.sav"))
lung_cancer_model = joblib.load(os.path.join(BASE_DIR, 'models', 'lung_cancer_model.sav'))
breast_cancer_model = joblib.load(os.path.join(BASE_DIR, 'models', 'breast_cancer.sav'))
chronic_disease_model = joblib.load(os.path.join(BASE_DIR, 'models', 'chronic_model.sav'))
hepatitis_model = joblib.load(os.path.join(BASE_DIR, 'models', 'hepititisc_model.sav'))
liver_model = joblib.load(os.path.join(BASE_DIR, 'models', 'liver_model.sav'))

# sidebar
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction', [
        'Disease Prediction',
        'Diabetes Prediction',
        'Heart disease Prediction',
        'Parkison Prediction',
        'Liver prediction',
        'Hepatitis prediction',
        'Lung Cancer Prediction',
        'Chronic Kidney prediction',
        'Breast Cancer Prediction',
    ],
        icons=['','activity', 'heart', 'person','person','person','person','bar-chart-fill'],
        default_index=0)


# multiple disease prediction
if selected == 'Disease Prediction': 
    # Create disease class and load ML model
    disease_model = DiseaseModel()
    # Assuming your xgboost model is in a folder named 'model' (singular). Change to 'models' if needed.
    disease_model.load_xgboost(os.path.join(BASE_DIR, 'model', 'xgboost_model.json'))

    # Title
    st.write('# Multistage classification of medical symptoms')

    symptoms = st.multiselect('What are your symptoms?', options=disease_model.all_symptoms)

    X = prepare_symptoms_array(symptoms)

    # Trigger XGBoost model
    if st.button('Predict'): 
        # Run the model with the python script
        prediction, prob = disease_model.predict(X)
        st.write(f'## Disease: {prediction} with {prob*100:.2f}% probability')

        tab1, tab2= st.tabs(["Description", "Precautions"])

        with tab1:
            st.write(disease_model.describe_predicted_disease())

        with tab2:
            precautions = disease_model.predicted_disease_precautions()
            for i in range(4):
                st.write(f'{i+1}. {precautions[i]}')


# Diabetes prediction page
if selected == 'Diabetes Prediction':  # pagetitle
    st.title("Diabetes disease prediction")
    image = Image.open(os.path.join(BASE_DIR, 'd3.jpg'))
    st.image(image, caption='diabetes disease prediction')
    
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input("Number of Pregnencies")
    with col2:
        Glucose = st.number_input("Glucose level")
    with col3:
        BloodPressure = st.number_input("Blood pressure  value")
    with col1:
        SkinThickness = st.number_input("Sckinthickness value")
    with col2:
        Insulin = st.number_input("Insulin value ")
    with col3:
        BMI = st.number_input("BMI value")
    with col1:
        DiabetesPedigreefunction = st.number_input("Diabetespedigreefunction value")
    with col2:
        Age = st.number_input("AGE")

    diabetes_dig = ''

    if st.button("Diabetes test result"):
        diabetes_prediction = diabetes_model.predict(
            [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreefunction, Age]])

        if diabetes_prediction[0] == 1:
            diabetes_dig = "we are really sorry to say but it seems like you are Diabetic."
            image = Image.open(os.path.join(BASE_DIR, 'positive.jpg'))
            st.image(image, caption='')
        else:
            diabetes_dig = 'Congratulation,You are not diabetic'
            image = Image.open(os.path.join(BASE_DIR, 'negative.jpg'))
            st.image(image, caption='')
        st.success(name+' , ' + diabetes_dig)
        

# Heart prediction page
if selected == 'Heart disease Prediction':
    st.title("Heart disease prediction")
    image = Image.open(os.path.join(BASE_DIR, 'heart2.jpg'))
    st.image(image, caption='heart failuire')
    
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age")
    with col2:
        sex=0
        display = ("male", "female")
        options = list(range(len(display)))
        value = st.selectbox("Gender", options, format_func=lambda x: display[x])
        # FIXED: value is an integer, so we check display[value]
        if display[value] == "male":
            sex = 1
        elif display[value] == "female":
            sex = 0
    with col3:
        cp=0
        display = ("typical angina","atypical angina","non — anginal pain","asymptotic")
        options = list(range(len(display)))
        value = st.selectbox("Chest_Pain Type", options, format_func=lambda x: display[x])
        if display[value] == "typical angina":
            cp = 0
        elif display[value] == "atypical angina":
            cp = 1
        elif display[value] == "non — anginal pain":
            cp = 2
        elif display[value] == "asymptotic":
            cp = 3
    with col1:
        trestbps = st.number_input("Resting Blood Pressure")
    with col2:
        chol = st.number_input("Serum Cholestrol")
    with col3:
        restecg=0
        display = ("normal","having ST-T wave abnormality","left ventricular hyperthrophy")
        options = list(range(len(display)))
        value = st.selectbox("Resting ECG", options, format_func=lambda x: display[x])
        if display[value] == "normal":
            restecg = 0
        elif display[value] == "having ST-T wave abnormality":
            restecg = 1
        elif display[value] == "left ventricular hyperthrophy":
            restecg = 2
    with col1:
        exang=0
        thalach = st.number_input("Max Heart Rate Achieved")
    with col2:
        oldpeak = st.number_input("ST depression induced by exercise relative to rest")
    with col3:
        slope=0
        display = ("upsloping","flat","downsloping")
        options = list(range(len(display)))
        value = st.selectbox("Peak exercise ST segment", options, format_func=lambda x: display[x])
        if display[value] == "upsloping":
            slope = 0
        elif display[value] == "flat":
            slope = 1
        elif display[value] == "downsloping":
            slope = 2
    with col1:
        ca = st.number_input("Number of major vessels (0–3) colored by flourosopy")
    with col2:
        thal=0
        display = ("normal","fixed defect","reversible defect")
        options = list(range(len(display)))
        value = st.selectbox("thalassemia", options, format_func=lambda x: display[x])
        if display[value] == "normal":
            thal = 0
        elif display[value] == "fixed defect":
            thal = 1
        elif display[value] == "reversible defect":
            thal = 2
    with col3:
        agree = st.checkbox('Exercise induced angina')
        if agree:
            exang = 1
        else:
            exang=0
    with col1:
        agree1 = st.checkbox('fasting blood sugar > 120mg/dl')
        if agree1:
            fbs = 1
        else:
            fbs=0

    heart_dig = ''

    if st.button("Heart test result"):
        heart_prediction = heart_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        if heart_prediction[0] == 1:
            heart_dig = 'we are really sorry to say but it seems like you have Heart Disease.'
            image = Image.open(os.path.join(BASE_DIR, 'positive.jpg'))
            st.image(image, caption='')
        else:
            heart_dig = "Congratulation , You don't have Heart Disease."
            image = Image.open(os.path.join(BASE_DIR, 'negative.jpg'))
            st.image(image, caption='')
        st.success(name +' , ' + heart_dig)


if selected == 'Parkison Prediction':
    st.title("Parkison prediction")
    image = Image.open(os.path.join(BASE_DIR, 'p1.jpg'))
    st.image(image, caption='parkinsons disease')

    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)
    with col1:
        MDVP = st.number_input("MDVP:Fo(Hz)")
    with col2:
        MDVPFIZ = st.number_input("MDVP:Fhi(Hz)")
    with col3:
        MDVPFLO = st.number_input("MDVP:Flo(Hz)")
    with col1:
        MDVPJITTER = st.number_input("MDVP:Jitter(%)")
    with col2:
        MDVPJitterAbs = st.number_input("MDVP:Jitter(Abs)")
    with col3:
        MDVPRAP = st.number_input("MDVP:RAP")
    with col2:
        MDVPPPQ = st.number_input("MDVP:PPQ ")
    with col3:
        JitterDDP = st.number_input("Jitter:DDP")
    with col1:
        MDVPShimmer = st.number_input("MDVP:Shimmer")
    with col2:
        MDVPShimmer_dB = st.number_input("MDVP:Shimmer(dB)")
    with col3:
        Shimmer_APQ3 = st.number_input("Shimmer:APQ3")
    with col1:
        ShimmerAPQ5 = st.number_input("Shimmer:APQ5")
    with col2:
        MDVP_APQ = st.number_input("MDVP:APQ")
    with col3:
        ShimmerDDA = st.number_input("Shimmer:DDA")
    with col1:
        NHR = st.number_input("NHR")
    with col2:
        HNR = st.number_input("HNR")
    with col2:
        RPDE = st.number_input("RPDE")
    with col3:
        DFA = st.number_input("DFA")
    with col1:
        spread1 = st.number_input("spread1")
    with col1:
        spread2 = st.number_input("spread2")
    with col3:
        D2 = st.number_input("D2")
    with col1:
        PPE = st.number_input("PPE")

    parkinson_dig = ''
    
    if st.button("Parkinson test result"):
        parkinson_prediction = parkinson_model.predict([[MDVP, MDVPFIZ, MDVPFLO, MDVPJITTER, MDVPJitterAbs, MDVPRAP, MDVPPPQ, JitterDDP, MDVPShimmer,MDVPShimmer_dB, Shimmer_APQ3, ShimmerAPQ5, MDVP_APQ, ShimmerDDA, NHR, HNR,  RPDE, DFA, spread1, spread2, D2, PPE]])

        if parkinson_prediction[0] == 1:
            parkinson_dig = 'we are really sorry to say but it seems like you have Parkinson disease'
            image = Image.open(os.path.join(BASE_DIR, 'positive.jpg'))
            st.image(image, caption='')
        else:
            parkinson_dig = "Congratulation , You don't have Parkinson disease"
            image = Image.open(os.path.join(BASE_DIR, 'negative.jpg'))
            st.image(image, caption='')
        st.success(name+' , ' + parkinson_dig)


# Load the dataset
lung_cancer_data = pd.read_csv(os.path.join(BASE_DIR, 'data', 'lung_cancer.csv'))

# Convert 'M' to 0 and 'F' to 1 in the 'GENDER' column
lung_cancer_data['GENDER'] = lung_cancer_data['GENDER'].map({'M': 'Male', 'F': 'Female'})

# Lung Cancer prediction page
if selected == 'Lung Cancer Prediction':
    st.title("Lung Cancer Prediction")
    image = Image.open(os.path.join(BASE_DIR, 'h.png'))
    st.image(image, caption='Lung Cancer Prediction')

    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender:", lung_cancer_data['GENDER'].unique())
    with col2:
        age = st.number_input("Age")
    with col3:
        smoking = st.selectbox("Smoking:", ['NO', 'YES'])
    with col1:
        yellow_fingers = st.selectbox("Yellow Fingers:", ['NO', 'YES'])
    with col2:
        anxiety = st.selectbox("Anxiety:", ['NO', 'YES'])
    with col3:
        peer_pressure = st.selectbox("Peer Pressure:", ['NO', 'YES'])
    with col1:
        chronic_disease = st.selectbox("Chronic Disease:", ['NO', 'YES'])
    with col2:
        fatigue = st.selectbox("Fatigue:", ['NO', 'YES'])
    with col3:
        allergy = st.selectbox("Allergy:", ['NO', 'YES'])
    with col1:
        wheezing = st.selectbox("Wheezing:", ['NO', 'YES'])
    with col2:
        alcohol_consuming = st.selectbox("Alcohol Consuming:", ['NO', 'YES'])
    with col3:
        coughing = st.selectbox("Coughing:", ['NO', 'YES'])
    with col1:
        shortness_of_breath = st.selectbox("Shortness of Breath:", ['NO', 'YES'])
    with col2:
        swallowing_difficulty = st.selectbox("Swallowing Difficulty:", ['NO', 'YES'])
    with col3:
        chest_pain = st.selectbox("Chest Pain:", ['NO', 'YES'])

    cancer_result = ''

    if st.button("Predict Lung Cancer"):
        user_data = pd.DataFrame({
            'GENDER': [gender],
            'AGE': [age],
            'SMOKING': [smoking],
            'YELLOW_FINGERS': [yellow_fingers],
            'ANXIETY': [anxiety],
            'PEER_PRESSURE': [peer_pressure],
            'CHRONICDISEASE': [chronic_disease],
            'FATIGUE': [fatigue],
            'ALLERGY': [allergy],
            'WHEEZING': [wheezing],
            'ALCOHOLCONSUMING': [alcohol_consuming],
            'COUGHING': [coughing],
            'SHORTNESSOFBREATH': [shortness_of_breath],
            'SWALLOWINGDIFFICULTY': [swallowing_difficulty],
            'CHESTPAIN': [chest_pain]
        })

        user_data.replace({'NO': 1, 'YES': 2}, inplace=True)
        user_data.columns = user_data.columns.str.strip()
        numeric_columns = ['AGE', 'FATIGUE', 'ALLERGY', 'ALCOHOLCONSUMING', 'COUGHING', 'SHORTNESSOFBREATH']
        user_data[numeric_columns] = user_data[numeric_columns].apply(pd.to_numeric, errors='coerce')

        cancer_prediction = lung_cancer_model.predict(user_data)

        if cancer_prediction[0] == 'YES':
            cancer_result = "The model predicts that there is a risk of Lung Cancer."
            image = Image.open(os.path.join(BASE_DIR, 'positive.jpg'))
            st.image(image, caption='')
        else:
            cancer_result = "The model predicts no significant risk of Lung Cancer."
            image = Image.open(os.path.join(BASE_DIR, 'negative.jpg'))
            st.image(image, caption='')

        st.success(name + ', ' + cancer_result)


# Liver prediction page
if selected == 'Liver prediction':  # pagetitle
    st.title("Liver disease prediction")
    image = Image.open(os.path.join(BASE_DIR, 'liver.jpg'))
    st.image(image, caption='Liver disease prediction.')

    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        Sex=0
        display = ("male", "female")
        options = list(range(len(display)))
        value = st.selectbox("Gender", options, format_func=lambda x: display[x])
        # FIXED: value is an integer, so we check display[value]
        if display[value] == "male":
            Sex = 0
        elif display[value] == "female":
            Sex = 1
    with col2:
        age = st.number_input("Entre your age") # 2 
    with col3:
        Total_Bilirubin = st.number_input("Entre your Total_Bilirubin") # 3
    with col1:
        Direct_Bilirubin = st.number_input("Entre your Direct_Bilirubin")# 4
    with col2:
        Alkaline_Phosphotase = st.number_input("Entre your Alkaline_Phosphotase") # 5
    with col3:
        Alamine_Aminotransferase = st.number_input("Entre your Alamine_Aminotransferase") # 6
    with col1:
        Aspartate_Aminotransferase = st.number_input("Entre your Aspartate_Aminotransferase") # 7
    with col2:
        Total_Protiens = st.number_input("Entre your Total_Protiens")# 8
    with col3:
        Albumin = st.number_input("Entre your Albumin") # 9
    with col1:
        Albumin_and_Globulin_Ratio = st.number_input("Entre your Albumin_and_Globulin_Ratio") # 10 

    liver_dig = ''

    if st.button("Liver test result"):
        liver_prediction = liver_model.predict([[Sex,age,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio]])

        if liver_prediction[0] == 1:
            image = Image.open(os.path.join(BASE_DIR, 'positive.jpg'))
            st.image(image, caption='')
            liver_dig = "we are really sorry to say but it seems like you have liver disease."
        else:
            image = Image.open(os.path.join(BASE_DIR, 'negative.jpg'))
            st.image(image, caption='')
            liver_dig = "Congratulation , You don't have liver disease."
        st.success(name+' , ' + liver_dig)


# Hepatitis prediction page
if selected == 'Hepatitis prediction':
    st.title("Hepatitis Prediction")
    image = Image.open(os.path.join(BASE_DIR, 'h.png'))
    st.image(image, caption='Hepatitis Prediction')

    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Enter your age")  # 2
    with col2:
        sex = st.selectbox("Gender", ["Male", "Female"])
        sex = 1 if sex == "Male" else 2
    with col3:
        total_bilirubin = st.number_input("Enter your Total Bilirubin")  # 3

    with col1:
        direct_bilirubin = st.number_input("Enter your Direct Bilirubin")  # 4
    with col2:
        alkaline_phosphatase = st.number_input("Enter your Alkaline Phosphatase")  # 5
    with col3:
        alamine_aminotransferase = st.number_input("Enter your Alamine Aminotransferase")  # 6

    with col1:
        aspartate_aminotransferase = st.number_input("Enter your Aspartate Aminotransferase")  # 7
    with col2:
        total_proteins = st.number_input("Enter your Total Proteins")  # 8
    with col3:
        albumin = st.number_input("Enter your Albumin")  # 9

    with col1:
        albumin_and_globulin_ratio = st.number_input("Enter your Albumin and Globulin Ratio")  # 10

    with col2:
        your_ggt_value = st.number_input("Enter your GGT value")  # Add this line
    with col3:
        your_prot_value = st.number_input("Enter your PROT value")  # Add this line

    hepatitis_result = ''

    if st.button("Predict Hepatitis"):
        user_data = pd.DataFrame({
            'Age': [age],
            'Sex': [sex],
            'ALB': [total_bilirubin],  
            'ALP': [direct_bilirubin],  
            'ALT': [alkaline_phosphatase],  
            'AST': [alamine_aminotransferase],
            'BIL': [aspartate_aminotransferase],  
            'CHE': [total_proteins],  
            'CHOL': [albumin],  
            'CREA': [albumin_and_globulin_ratio],  
            'GGT': [your_ggt_value],  
            'PROT': [your_prot_value]  
        })
        
        # Note: The mapping above looks suspicious (e.g., mapping Total Bilirubin to ALB). 
        # Ensure the DataFrame columns match exactly what the hepatitis_model was trained on.
        # You may need to add a prediction line here like: hepatitis_prediction = hepatitis_model.predict(user_data)
