import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="wide"
)

# --- Helper Functions ---
@st.cache_resource
def load_model():
    model_path = os.path.join("models", "student_model.joblib")
    if not os.path.exists(model_path):
        st.error(f"Model file not found at '{model_path}'. Please run `train_model.py` first!")
        st.stop()
    return joblib.load(model_path)

@st.cache_data
def load_data():
    data_path = os.path.join("data", "student-mat.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path, sep=";")
    return None

# Load model and raw data
model_pipeline = load_model()
df_raw = load_data()

# --- Main Interface ---
st.title("🎓 Student Performance Prediction System")
st.markdown("""
Welcome to the ML-driven Student Grade Prediction Platform. 
Predict student final grades (**G3**, scale 0–20) based on demographic, social, and academic period features.
""")

tab1, tab2 = st.tabs(["🔮 Single Student Prediction", "📊 Dataset Insights & EDA"])

# ==========================================
# TAB 1: Real-Time Prediction Form
# ==========================================
with tab1:
    st.subheader("Enter Student Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📚 Academic Context")
        g1 = st.slider("First Period Grade (G1)", 0, 20, 10)
        g2 = st.slider("Second Period Grade (G2)", 0, 20, 10)
        failures = st.selectbox("Past Class Failures", [0, 1, 2, 3])
        studytime = st.selectbox("Weekly Study Time (hours)", 
                                 options=[1, 2, 3, 4], 
                                 format_func=lambda x: {1: "< 2 hours", 2: "2 to 5 hours", 3: "5 to 10 hours", 4: "> 10 hours"}[x])
        absences = st.slider("Number of Absences", 0, 93, 4)
        schoolsup = st.selectbox("Extra Educational Support", ["yes", "no"])
        famsup = st.selectbox("Family Educational Support", ["yes", "no"])
        paid = st.selectbox("Extra Paid Classes", ["yes", "no"])

    with col2:
        st.markdown("### 🧑‍🤝‍🧑 Lifestyle & Social Factors")
        age = st.slider("Age", 15, 22, 16)
        sex = st.selectbox("Sex", ["F", "M"])
        address = st.selectbox("Address Type", ["U", "R"], format_func=lambda x: "Urban" if x == "U" else "Rural")
        goout = st.slider("Going Out with Friends (1: Very Low to 5: Very High)", 1, 5, 3)
        dalc = st.slider("Workday Alcohol Consumption (1-5)", 1, 5, 1)
        walc = st.slider("Weekend Alcohol Consumption (1-5)", 1, 5, 1)
        health = st.slider("Current Health Status (1: Very Bad to 5: Very Good)", 1, 5, 4)
        higher = st.selectbox("Wants to Pursue Higher Education", ["yes", "no"])

    with col3:
        st.markdown("### 🏠 Family & Background")
        medu = st.selectbox("Mother's Education", [0, 1, 2, 3, 4], 
                             format_func=lambda x: {0: "None", 1: "Primary (4th grade)", 2: "5th to 9th grade", 3: "Secondary", 4: "Higher Education"}[x])
        fedu = st.selectbox("Father's Education", [0, 1, 2, 3, 4], 
                             format_func=lambda x: {0: "None", 1: "Primary (4th grade)", 2: "5th to 9th grade", 3: "Secondary", 4: "Higher Education"}[x])
        mjob = st.selectbox("Mother's Job", ["teacher", "health", "services", "at_home", "other"])
        fjob = st.selectbox("Father's Job", ["teacher", "health", "services", "at_home", "other"])
        famsize = st.selectbox("Family Size", ["LE3", "GT3"], format_func=lambda x: "≤ 3 members" if x == "LE3" else "> 3 members")
        pstatus = st.selectbox("Parents Cohabitation Status", ["T", "A"], format_func=lambda x: "Living Together" if x == "T" else "Apart")
        school = st.selectbox("School", ["GP", "MS"], format_func=lambda x: "Gabriel Pereira" if x == "GP" else "Mousinho da Silveira")
        reason = st.selectbox("Reason to Choose School", ["home", "reputation", "course", "other"])
        guardian = st.selectbox("Guardian", ["mother", "father", "other"])
        activities = st.selectbox("Extra-curricular Activities", ["yes", "no"])
        nursery = st.selectbox("Attended Nursery School", ["yes", "no"])
        internet = st.selectbox("Internet Access at Home", ["yes", "no"])
        romantic = st.selectbox("In a Romantic Relationship", ["yes", "no"])
        freetime = st.slider("Free Time After School (1-5)", 1, 5, 3)
        traveltime = st.selectbox("Home to School Travel Time", [1, 2, 3, 4], 
                                  format_func=lambda x: {1: "< 15 min", 2: "15-30 min", 3: "30 min-1 hour", 4: "> 1 hour"}[x])

    st.markdown("---")
    
    if st.button("🚀 Predict Final Grade (G3)", use_container_width=True):
        # Create input dataframe matching feature names
        input_data = pd.DataFrame([{
            "school": school, "sex": sex, "age": age, "address": address, "famsize": famsize,
            "Pstatus": pstatus, "Medu": medu, "Fedu": fedu, "Mjob": mjob, "Fjob": fjob,
            "reason": reason, "guardian": guardian, "traveltime": traveltime, "studytime": studytime,
            "failures": failures, "schoolsup": schoolsup, "famsup": famsup, "paid": paid,
            "activities": activities, "nursery": nursery, "higher": higher, "internet": internet,
            "romantic": romantic, "famrel": 4, "freetime": freetime, "goout": goout,
            "Dalc": dalc, "Walc": walc, "health": health, "absences": absences,
            "G1": g1, "G2": g2
        }])

        # Predict
        prediction = model_pipeline.predict(input_data)[0]
        final_score = np.clip(round(prediction, 2), 0, 20)

        # Output Card
        st.markdown("### 🎯 Prediction Results")
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric(label="Predicted Final Grade (G3)", value=f"{final_score} / 20")
            
        with res_col2:
            if final_score >= 10:
                st.success("✅ **Status:** Passed (Eligible to proceed)")
            else:
                st.error("⚠️ **Status:** At-Risk (Intervention recommended)")

# ==========================================
# TAB 2: Dataset Visual Insights
# ==========================================
with tab2:
    st.subheader("Data Exploratory Overview")
    if df_raw is not None:
        fig_hist = px.histogram(df_raw, x="G3", nbins=20, title="Distribution of Final Grades (G3)", color_discrete_sequence=['teal'])
        st.plotly_chart(fig_hist, use_container_width=True)
        
        fig_box = px.box(df_raw, x="failures", y="G3", title="Impact of Past Failures on Final Grade", color="failures")
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("Raw dataset not available to render exploratory charts.")