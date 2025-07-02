import streamlit as st
import pandas as pd
import pickle
from utils.preprocess import load_and_clean_data, encode_input

# ✅ This must be first!
st.set_page_config(page_title="IncomeInsight", layout="wide")

# -------------------------------
# 🔃 Load model and data
# -------------------------------

@st.cache_resource
def load_model():
    with open("model/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    with open("model/columns.pkl", "rb") as f:
        column_order = pickle.load(f)
    return model, encoders, column_order

@st.cache_data
def get_dataset():
    return load_and_clean_data("data/adult.csv")

model, encoders, column_order = load_model()
data = get_dataset()

# -------------------------------
# 🚪 Sidebar Navigation
# -------------------------------

page = st.sidebar.radio("📌 Navigate", ["🏠 Home", "🔍 Explore Data", "📊 Feature Importance", "🤖 Make Prediction"])

# -------------------------------
# 🏠 Home Page
# -------------------------------

if page == "🏠 Home":
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🚀 Welcome to IncomeInsight</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: gray;'>Predict whether a person's income exceeds $50K/year using machine learning</h4>", unsafe_allow_html=True)

    st.image("https://cdn.pixabay.com/photo/2015/01/08/18/29/startup-593327_960_720.jpg", use_container_width=True)

    st.markdown("### 💡 Why This App?")
    st.info("""
    Every year, governments and industries collect census data to understand social and economic trends.
    This app uses machine learning to predict who earns more than $50K/year based on this data.
    """)

    with st.expander("📂 Dataset Used"):
        st.markdown("""
        The dataset is the **Adult Census Income Dataset** from the UCI ML Repository.  
        It contains demographic, education, work, and financial details of over 32,000 individuals.
        """)

    with st.expander("⚙️ Technologies Behind This App"):
        st.markdown("""
        - 🧠 **Machine Learning**: Random Forest Classifier  
        - 🧮 **Data Handling**: pandas, scikit-learn  
        - 🎨 **Interface**: Streamlit  
        - 🗂️ **Structure**: Modular project with preprocessing and model separation
        """)

    with st.expander("🧪 What You Can Do Here"):
        st.markdown("""
        - 🔍 Explore the dataset  
        - 📊 Analyze feature importance  
        - 🤖 Make income predictions with live form  
        - 📈 See prediction confidence
        """)

    st.success("👉 Use the sidebar to begin!")

# -------------------------------
# 🔍 Explore Data Page
# -------------------------------

elif page == "🔍 Explore Data":
    st.title("🔍 Explore the Census Dataset")
    st.write(f"Total Records: {len(data)}")
    st.dataframe(data.head())

    if st.checkbox("Show summary statistics"):
        st.write(data.describe())

    if st.checkbox("Show income distribution"):
        st.bar_chart(data['income'].value_counts())

# -------------------------------
# 📊 Feature Importance Page
# -------------------------------

elif page == "📊 Feature Importance":
    st.title("📊 Feature Importance from Random Forest")
    importance = pd.Series(model.feature_importances_, index=column_order)
    st.bar_chart(importance.sort_values(ascending=True))

# -------------------------------
# 🤖 Make Prediction Page
# -------------------------------

elif page == "🤖 Make Prediction":
    st.title("🤖 Predict Income > $50K/year")

    with st.form("prediction_form"):
        st.subheader("Enter Person Details")

        # Numeric features
        age = st.slider("Age", 17, 90, 35)
        fnlwgt = st.number_input("Final weight", 10000, 1000000, 190000)
        education_num = st.slider("Education Num", 1, 16, 10)
        capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
        capital_loss = st.number_input("Capital Loss", 0, 5000, 0)
        hours_per_week = st.slider("Hours per Week", 1, 99, 40)

        # Categorical features
        workclass = st.selectbox("Workclass", encoders['workclass'].classes_)
        education = st.selectbox("Education", encoders['education'].classes_)
        marital_status = st.selectbox("Marital Status", encoders['marital_status'].classes_)
        occupation = st.selectbox("Occupation", encoders['occupation'].classes_)
        relationship = st.selectbox("Relationship", encoders['relationship'].classes_)
        race = st.selectbox("Race", encoders['race'].classes_)
        sex = st.selectbox("Sex", encoders['sex'].classes_)
        native_country = st.selectbox("Native Country", encoders['native_country'].classes_)

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = pd.DataFrame([{ 
            'age': age,
            'fnlwgt': fnlwgt,
            'education_num': education_num,
            'capital_gain': capital_gain,
            'capital_loss': capital_loss,
            'hours_per_week': hours_per_week,
            'workclass': workclass,
            'education': education,
            'marital_status': marital_status,
            'occupation': occupation,
            'relationship': relationship,
            'race': race,
            'sex': sex,
            'native_country': native_country
        }])

        # Encode and reorder
        input_encoded = encode_input(input_data, encoders)
        input_encoded = input_encoded[column_order]

        # Predict
        prediction = model.predict(input_encoded)
        proba = model.predict_proba(input_encoded)[0][1]

        if prediction[0] == 1:
            st.success(f"✅ Prediction: Income > $50K/year (Confidence: {proba:.2%})")
        else:
            st.error(f"❌ Prediction: Income ≤ $50K/year (Confidence: {1 - proba:.2%})")
