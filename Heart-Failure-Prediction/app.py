# import necessary libraries
import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import base64



# Create a function to download binary files
def html_binary_file_downloader(df):
    csv = df.to_csv(index=False)
    encoder_decoder = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{encoder_decoder}" download="predictions.csv">Download Predictions CSV</a>'
    return href


# Create App Name and Tabs
st.title('Heart Disease Predictor')
tab1, tab2, tab3 = st.tabs(['Predict', 'Batch Prediction', 'Model Information'])


# Add features to each Tab
with tab1:
    age = st.number_input("Age (in years)", min_value=0, max_value=135)
    sex = st.selectbox("Sex", ['Male', 'Female'])
    chest_pain = st.selectbox("Chest Pain Type", ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asympyomatic'])
    resting_bp = st.number_input("Resting Blood Pressure (in mmHg)", min_value=0, max_value=300)
    cholesterol = st.number_input("Serum Cholesterol (mm/dl)", min_value=0)
    fasting_bs = st.selectbox("Fasting Blood Sugar", ["<=120 mg/dl", "> 120 mg/dl"])
    resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=120)
    exercise_angina = st.selectbox("Exercise-induced Angina", ["Yes", "No"])
    old_peak = st.number_input("ST Depression", min_value = 0.0, max_value=10.0)
    st_slope = st.selectbox("Slope of the peak exercise ST segment", ["Upsloping", "Flat", "Downsloping"])


    # Convert all Categorical Columns to Numeric
    sex = 0 if sex == "Male" else 1
    chest_pain = ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asympyomatic'].index(chest_pain)
    fasting_bs = 1 if fasting_bs == "> 120 mg/dl" else 0
    resting_ecg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg)
    exercise_angina = 1 if exercise_angina == "Yes" else 0
    st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope)


    # Create a DataFrame with user inputs
    input = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType':[chest_pain],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG': [resting_ecg],
        'MaxHR':[max_hr],
        'ExerciseAngina': [exercise_angina],
        'Oldpeak' : [old_peak],
        'ST_Slope' : [st_slope]
    })

    models = ['Random Forest', 'Support Vector Machine', 'Decision Trees', 'Logistic Regression']
    model_names = ['randomforest.pkl', 'svm.pkl', 'decisiontree.pkl', 'logisticregression.pkl']


    predictions = []
    def heart_disease_prediction(data):
        for model_name in model_names:
            model = pickle.load(open(model_name, 'rb'))
            prediction = model.predict(data)
            predictions.append(prediction)
        return predictions
    

    # Activate the submit button for prediction generation
    if st.button("Submit"):
        st.subheader('Loading Results...')
        st.markdown('*******************************************')

        result = heart_disease_prediction(input)


        for i in range(len(predictions)):
            st.subheader(models[i])
            if result[i][0] == 0:
                st.write("No heart disease has been detected")
            else:
                st.write("Heart disease has been detected")

            st.markdown('*******************************************')

# Batch Prediction
with tab2:
    st.title("Upload a CSV file")
    st.subheader("Kindly take note of the following before uploading your file:")
    st.info("""
        - No null values are accepted. You are to submit a cleaned data
        - There must be 11 features in total, as used in training the model
        - Ensure correct spellings of feature names
        -Ensure conventional pattern of feature values, that is,
            1. Age:  age of the patient in years
            2. Sex: sex of the patient( 1 for female and 0 for male)
            3. ChestPainType: (0:Atypical Angina, 1:Non-Anginal Pain, 2:Asymptomatic, 3:Typical Angina)
            4. RestingBP: (Resting Blood Pressure measured in mmHg)
            5. Cholesterol: Serum Cholesterol (mm/dl)
            6. Fasting_BS: Fasting Blood Sugar [0: if <=120 mg/dl", 1: if> 120 mg/dl]
            7. RestingECG: Resting Electrocardiogram results [0:Normal, 1: ST-T Wave Abnormality, 2:Left Ventricular Hypertrophy]
            8. MaxHR: Maximum Heart Rate Achieved (numeric value between 60 and 120)
            9. ExerciseAngina: Exercise-induced Angina [1:Yes, 0:No]
            10 Oldpeak : ST numeric value measured in Depression
            11. ST_Slope: Slope of the peak exercise ST segment [0:Upsloping, 1:Flat, 2:Downsloping]   
        """)

    file_upload = st.file_uploader('Upload a CSV file', type=['csv'])
    
    if file_upload is not None:
        input = pd.read_csv(file_upload)
        model = pickle.load(open('logisticregression.pkl', 'rb'))

        # Create DataFrame column
        dataframe_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

        if set(dataframe_columns).issubset(input.columns):

            # input['Prediction'] = lambda x: 'Heart Disease Predicted' if model.predict(input) == 0 else 'Heart Disease Not Predicted'

            # input['Prediction'] = model.predict(input).map({1: 'Heart Disease Predicted', 0: 'Heart Disease Not Predicted'})
            # input['Prediction'] = map({1: 'Heart Disease Predicted', 0: 'Heart Disease Not Predicted'}, model.predict(input))
            input['Prediction'] = model.predict(input)
            input['Prediction'] = input['Prediction'].map({1: 'Heart Disease Predicted', 0: 'Heart Disease Not Predicted'})

            # for i in range(len(input)):
            #     array = input.iloc[i, :-1].values.reshape(1, -1)
            #     input.loc[i, 'Prediction'][i] = model.predict(array)[0]
            input.to_csv('Predicted_Result.csv')

                # Deploy the predictions
            st.subheader('Predictions')
            st.write(input)


            # Create button to download updated csv file
            st.markdown(html_binary_file_downloader(input), unsafe_allow_html=True)
        else:
            st.warnings('Ensure uploaded files has the correct dataframe_columns')

    else:
        st.info("Please upload a valid CSV file to get your predictions")

with tab3:
    data = {'Logistic Regression':85.33, 'Support Vector Machine':86.92, 'Decision Tree': 90.69, 'Random Forest': 86.00}
    Algorithms = list(data.keys())
    Accuracy_Scores = list(data.values())
    df = pd.DataFrame(list(zip(Algorithms, Accuracy_Scores)), columns= ['Algorithms', 'Accuracy_Scores'])

    fig = px.bar(df, y='Accuracy_Scores', x='Algorithms', color='Algorithms')
    st.plotly_chart(fig)
