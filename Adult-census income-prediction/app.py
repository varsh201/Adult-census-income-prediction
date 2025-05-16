import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Set up the app
st.set_page_config(page_title="Census Income Prediction", layout="wide")

# Title
st.title("Census Income Prediction")
st.write("This app predicts if a person's income exceeds $50K/year based on census data.")

# Navigation
page = st.sidebar.radio("Go to", ["Introduction", "Data Exploration", "Model Training", "Make Prediction"])

# Load data
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num',
        'marital_status', 'occupation', 'relationship', 'race', 'sex',
        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
    ]
    data = pd.read_csv(url, header=None, names=column_names, na_values=' ?', skipinitialspace=True)
    data = data.dropna()
    data['income'] = data['income'].apply(lambda x: 1 if x == '>50K' else 0)
    return data

data = load_data()

# Introduction Page
if page == "Introduction":
    st.header("Welcome to Census Income Prediction")
    st.subheader("What is this application about?")
    st.write("This app demonstrates how machine learning can be used to predict income levels based on census data.")
    
    st.subheader("The dataset: Adult Census Income")
    st.write("The Adult Census Income dataset contains demographic information about individuals, such as:")
    st.write("- Age, education, occupation")
    st.write("- Marital status, race, gender")
    st.write("- Working hours, and more")
    st.write("The prediction task is to determine whether a person earns more than $50K per year.")

# Data Exploration Page
elif page == "Data Exploration":
    st.header("Data Exploration")
    st.subheader("Dataset Overview")
    st.write(f"Number of records: {len(data)}")
    st.write(f"Number of features: {len(data.columns)}")
    
    if st.checkbox("View first few rows of the dataset"):
        st.write(data.head())
    
    if st.checkbox("View dataset summary statistics"):
        st.write(data.describe())

# Model Training Page
elif page == "Model Training":
    st.header("Model Training")
    st.write("We'll train a Random Forest model to predict income.")
    
    # Prepare data
    X = data.drop('income', axis=1)
    y = data['income']
    
    # Convert categorical variables to numerical
    label_encoders = {}
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model and encoders
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    
    st.success("Model trained and saved successfully!")
    st.write(f"Training accuracy: {model.score(X_train, y_train):.2f}")
    st.write(f"Test accuracy: {model.score(X_test, y_test):.2f}")

# Prediction Page
elif page == "Make Prediction":
    st.header("Make Prediction")
    st.write("This form allows you to input custom values for each feature used by the model.")
    
    # Load model and encoders if they exist
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
    except:
        st.error("Please train the model first on the Model Training page.")
        st.stop()
    
    # Create form
    with st.form("prediction_form"):
        st.subheader("Numeric Features")
        age = st.slider("Age", 17, 90, 35)
        fnlwgt = st.number_input("Final weight (census expansion factor)", min_value=10000, max_value=1000000, value=189793)
        education_num = st.slider("Education (numeric scale)", 1, 16, 10)
        capital_gain = st.number_input("Capital gain ($)", min_value=0, max_value=100000, value=0)
        capital_loss = st.number_input("Capital loss ($)", min_value=0, max_value=5000, value=0)
        hours_per_week = st.slider("Hours per week", 1, 99, 40)
        
        st.subheader("Categorical Features")
        workclass = st.selectbox("Work Class", ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 
                                              'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
        education = st.selectbox("Education Level", ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 
                                                   'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', 
                                                   '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'])
        marital_status = st.selectbox("Marital Status", ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 
                                                        'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
        occupation = st.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 
                                               'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 
                                               'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 
                                               'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])
        relationship = st.selectbox("Relationship Status", ['Wife', 'Own-child', 'Husband', 'Not-in-family', 
                                                          'Other-relative', 'Unmarried'])
        race = st.selectbox("Race", ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
        sex = st.selectbox("Sex", ['Male', 'Female'])
        native_country = st.selectbox("Native Country", ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 
                                                        'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 
                                                        'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 
                                                        'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 
                                                        'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 
                                                        'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 
                                                        'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 
                                                        'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 
                                                        'Peru', 'Hong', 'Holand-Netherlands'])
        
        submitted = st.form_submit_button("Predict Income")
        
        if submitted:
            # Prepare input data with consistent column names
            input_data = {
                'age': age,
                'workclass': workclass,
                'fnlwgt': fnlwgt,
                'education': education,
                'education_num': education_num,
                'marital_status': marital_status,
                'occupation': occupation,
                'relationship': relationship,
                'race': race,
                'sex': sex,
                'capital_gain': capital_gain,
                'capital_loss': capital_loss,
                'hours_per_week': hours_per_week,
                'native_country': native_country
            }
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Encode categorical variables
            for column in input_df.select_dtypes(include=['object']).columns:
                le = label_encoders[column]  # Now using consistent column names
                input_df[column] = le.transform(input_df[column])
            
            # Make prediction
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)[0][1]
            
            # Show result
            if prediction[0] == 1:
                st.success(f"Prediction: Income > $50K/year (Probability: {probability:.2%})")
            else:
                st.success(f"Prediction: Income ≤ $50K/year (Probability: {1-probability:.2%})")