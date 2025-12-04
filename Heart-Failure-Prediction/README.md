
# Heart Disease Prediction (Machine Learning and Streamlit App)

This project delivers a full-cycle machine-learning solution for predicting heart disease using clinical features such as Age, Sex, Chest Pain Type, Cholesterol levels, Exercise Angina, ST Slope, and more. The solution ships with a fully deployed Streamlit web app for real-time inference and exploratory analysis.

## Project Overview

Cardiovascular disease is still the world’s most persistent productivity killer. This project builds a data-driven risk-prediction system using structured clinical data.
Key objectives include:

- Conducting EDA to uncover patterns across clinical features
- Modeling the likelihood of heart disease
- Deploying an interactive Streamlit interface
- Demonstrate how each feature influences model outputs using a sunburst visualization and other plots

## 🧬 Dataset Features

The dataset includes 12 core clinical variables:

|Feature|Description|
|----------|-------------|
|Age|Patient’s age|
|Sex|Male/Female|
|ChestPainType|Type of chest pain (ATA, NAP, TA, ASY)|
|RestingBP|Resting blood pressure|
|Cholesterol|Serum cholesterol level|
|FastingBS|Fasting blood sugar (1 = >120 mg/dl)|
|RestingECG|Resting ECG results|
|MaxHR|Maximum heart rate achieved|
|ExerciseAngina|Exercise-induced angina (Yes/No)|
|Oldpeak|ST depression induced by exercise|
|ST_Slope|Slope of the ST segment during exercise|
|HeartDisease|Target variable (1 = Yes, 0 = No)|

## 🎯 Key Deliverables

### **Exploratory Data Analysis (EDA)**

Understand distributions, correlations, and behavioral patterns across features.

### **Visual Insights**

- Sunburst chart for all features vs Heart Disease

- Correlation heatmap

- Class distribution plots

### **Machine Learning Pipeline**

- Data preprocessing

- Model training

- Feature importance ranking

- Performance evaluation

### **Deployment**

Streamlit app for real-time prediction and interactive visualization.

## ⚙️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-Learn
- Plotly / Matplotlib
- Streamlit
- Jupyter Notebook

## 🌐 Live Demo

heart-disease1.streamlit.app/

## 🤝 Contribution

Open to PRs and issue reports for improvements, new visualizations, or model enhancements.

## 📜 License

MIT License
