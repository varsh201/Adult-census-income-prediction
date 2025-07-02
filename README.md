# IncomeInsight 🧠💸

An interactive ML app that predicts if a person earns more than $50K/year based on U.S. Census data.

## 🔍 Features
- Explore real census data
- Train a Random Forest model
- Input form for real-time prediction
- Built using Streamlit + Scikit-learn

## 📦 Tech Stack
- Python
- Pandas
- Scikit-learn
- Streamlit

## 🚀 Run It
```bash
streamlit run app.py


---

## 🚀 How to Run This Application

Make sure you have **Python installed (>= 3.8)** and `pip` set up.

### 🔧 Step 1: Install the dependencies

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install streamlit pandas scikit-learn
```

---

### ▶️ Step 2: Run the Streamlit app

From the root directory of the project:

```bash
streamlit run app.py
```

It will automatically open in your browser at:

```
http://localhost:8501
```

---

### 📁 Project Folder Structure

```bash
IncomeInsight/
│
├── app.py                     # Main Streamlit app
├── requirements.txt           # Dependencies
│
├── data/
│   └── adult.csv              # Cleaned dataset
│
├── model/
│   ├── model.pkl              # Trained RandomForest model
│   ├── encoders.pkl           # Label encoders for categorical features
│   └── columns.pkl            # Column order used during training
│
└── utils/
    └── preprocess.py          # Data cleaning and encoding functions
```

---

Let me know if you want a full `README.md` file generated with badges, title, intro, usage, etc. — I can drop the full thing ready to upload to GitHub 💪
