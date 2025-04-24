import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data and scaler
@st.cache_data
def load_data():
    df = pd.read_csv('diabetes_data_upload.csv')
    return df

@st.cache_data
def load_scaler():
    return joblib.load('scaler.pkl')

df = load_data()
scaler = load_scaler()

# Sidebar navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ["Dataset Overview", "Data Visualization", "Diabetes Prediction"])

# Dataset Overview Page
if option == "Dataset Overview":
    st.title("Diabetes Dataset Overview")
    
    st.write("""
    This dataset contains information about patients and whether they have diabetes or not.
    The dataset includes various symptoms and demographic information.
    """)
    
    st.subheader("First 10 Rows of the Dataset")
    st.dataframe(df.head(10))
    
    st.subheader("Dataset Statistics")
    st.write(df.describe())
    
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

# Data Visualization Page
elif option == "Data Visualization":
    st.title("Data Visualization")
    
    st.write("Explore the relationships between different features and diabetes diagnosis.")
    
    # Age Distribution by Diabetes Status
    st.subheader("Age Distribution by Diabetes Status")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='class', y='Age', data=df, ax=ax)
    ax.set_title('Age Distribution by Diabetes Status')
    ax.set_xlabel('Diabetes Status')
    ax.set_ylabel('Age')
    ax.set_xticklabels(['Positive', 'Negative'])
    st.pyplot(fig)
    
    # Gender Distribution
    st.subheader("Gender Distribution by Diabetes Status")
    gender_counts = df.groupby(['Gender', 'class']).size().unstack()
    fig, ax = plt.subplots(figsize=(10, 6))
    gender_counts.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title('Gender Distribution by Diabetes Status')
    ax.set_xlabel('Gender')
    ax.set_ylabel('Count')
    ax.set_xticklabels(['Male', 'Female'], rotation=0)
    ax.legend(['Positive', 'Negative'])
    st.pyplot(fig)
    
    # Symptom Prevalence
    st.subheader("Symptom Prevalence in Diabetic Patients")
    symptoms = df.columns[2:-1]  # All symptom columns
    positive_cases = df[df['class'] == 'Positive']
    
    symptom_counts = positive_cases[symptoms].apply(lambda x: x.value_counts()).fillna(0)
    symptom_percent = (symptom_counts.loc['Yes'] / len(positive_cases)) * 100
    
    fig, ax = plt.subplots(figsize=(12, 8))
    symptom_percent.sort_values().plot(kind='barh', ax=ax)
    ax.set_title('Percentage of Diabetic Patients Reporting Each Symptom')
    ax.set_xlabel('Percentage (%)')
    ax.set_ylabel('Symptom')
    st.pyplot(fig)

# Prediction Page
elif option == "Diabetes Prediction":
    st.title("Diabetes Risk Prediction")
    
    st.write("""
    Fill in the form below to assess your risk of diabetes based on symptoms.
    All fields are required.
    """)
    
    # Create a form for user input
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=30)
            gender = st.radio("Gender", ["Male", "Female"])
            polyuria = st.radio("Polyuria (Excessive Urination)", ["No", "Yes"])
            polydipsia = st.radio("Polydipsia (Excessive Thirst)", ["No", "Yes"])
            sudden_weight_loss = st.radio("Sudden Weight Loss", ["No", "Yes"])
            weakness = st.radio("Weakness", ["No", "Yes"])
            polyphagia = st.radio("Polyphagia (Excessive Hunger)", ["No", "Yes"])
            
        with col2:
            genital_thrush = st.radio("Genital Thrush", ["No", "Yes"])
            visual_blurring = st.radio("Visual Blurring", ["No", "Yes"])
            itching = st.radio("Itching", ["No", "Yes"])
            irritability = st.radio("Irritability", ["No", "Yes"])
            delayed_healing = st.radio("Delayed Healing", ["No", "Yes"])
            partial_paresis = st.radio("Partial Paresis", ["No", "Yes"])
            muscle_stiffness = st.radio("Muscle Stiffness", ["No", "Yes"])
            alopecia = st.radio("Alopecia", ["No", "Yes"])
            obesity = st.radio("Obesity", ["No", "Yes"])
        
        submitted = st.form_submit_button("Predict Diabetes Risk")
    
    if submitted:
        # Prepare the input data
        input_data = {
            'Age': age,
            'Gender': 0 if gender == "Male" else 1,
            'Polyuria': 0 if polyuria == "Yes" else 1,
            'Polydipsia': 0 if polydipsia == "Yes" else 1,
            'sudden weight loss': 0 if sudden_weight_loss == "Yes" else 1,
            'weakness': 0 if weakness == "Yes" else 1,
            'Polyphagia': 0 if polyphagia == "Yes" else 1,
            'Genital thrush': 0 if genital_thrush == "Yes" else 1,
            'visual blurring': 0 if visual_blurring == "Yes" else 1,
            'Itching': 0 if itching == "Yes" else 1,
            'Irritability': 0 if irritability == "Yes" else 1,
            'delayed healing': 0 if delayed_healing == "Yes" else 1,
            'partial paresis': 0 if partial_paresis == "Yes" else 1,
            'muscle stiffness': 0 if muscle_stiffness == "Yes" else 1,
            'Alopecia': 0 if alopecia == "Yes" else 1,
            'Obesity': 0 if obesity == "Yes" else 1
        }
        
        # Convert to DataFrame and scale
        input_df = pd.DataFrame([input_data])
        scaled_input = scaler.transform(input_df)
        
        # Here you would normally make a prediction with your model
        # For now, we'll simulate a prediction (replace with your actual model)
        # prediction = model.predict(scaled_input)
        # probability = model.predict_proba(scaled_input)[0][1]
        
        # Simulated prediction (replace with your actual model)
        probability = np.random.uniform(0, 1)  # Replace with actual prediction
        prediction = "Positive" if probability > 0.5 else "Negative"
        
        # Display results
        st.subheader("Prediction Results")
        
        if prediction == "Positive":
            st.error(f"Prediction: {prediction} (Risk of Diabetes)")
            st.write(f"Probability: {probability*100:.2f}%")
            st.warning("""
            **Recommendations:**
            - Consult with a healthcare professional
            - Monitor your blood sugar levels
            - Maintain a healthy diet and exercise regularly
            """)
        else:
            st.success(f"Prediction: {prediction} (Low Risk of Diabetes)")
            st.write(f"Probability: {(1-probability)*100:.2f}%")
            st.info("""
            **Recommendations:**
            - Continue healthy lifestyle habits
            - Regular check-ups are still recommended
            - Be aware of diabetes symptoms
            """)