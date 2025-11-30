import streamlit as st
import joblib
import numpy as np

# Load model & encoder
model = joblib.load('return_risk_model.pkl')
le = joblib.load('category_encoder.pkl')

st.title("📦 E-Commerce Return Risk Predictor")
st.markdown("AI tool to flag high-risk orders before shipping")

order_value = st.number_input("Order Value ($)", min_value=10, max_value=500, value=100)
customer_age = st.slider("Customer Age", 18, 70, 30)
category = st.selectbox("Product Category", ['Electronics', 'Clothing', 'Home', 'Beauty'])
first_time = st.checkbox("First-Time Buyer?")

if st.button("Predict Return Risk"):
    cat_enc = le.transform([category])[0]
    features = np.array([[order_value, customer_age, cat_enc, int(first_time)]])
    prob = model.predict_proba(features)[0][1]  # Probability of return
    
    if prob > 0.5:
        st.error(f"⚠️ High Return Risk: {prob:.0%} chance")
    else:
        st.success(f"✅ Low Return Risk: {prob:.0%} chance")