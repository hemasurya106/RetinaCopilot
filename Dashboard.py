import streamlit as st

def run():
    st.title("Dashboard - AI-Powered Retinal Disease Grading Assistant")

    st.markdown("""
    ### 📌 Project Overview:
    This AI-powered assistant uses deep learning (ResNet50) to grade the severity of diabetic retinopathy from retinal fundus images.

    **Purpose:**  
    🔍 Early detection of diabetic eye diseases  
    💾 Built using the APTOS 2019 Blindness Detection dataset (for academic, non-commercial use only)

    ### 📊 Model Performance:
    - **Best Validation Accuracy:** 78.99% (ResNet50 with data augmentation and class balancing)
    - **Loss Function:** Cross Entropy with Class Weights
    - **Optimizer:** AdamW
    - **Epochs:** 30
    - **Scheduler:** StepLR (decay every 3 epochs)

    **Note:** This tool is for academic and research demonstration only.
    """)
    st.markdown("<h2 style='text-align: center;'>Diabetic Retinopathy</h2>", unsafe_allow_html=True)
    st.image("/Users/hemasurya/Desktop/RETINAL_PR/assets/image copy.png", caption="NORMAL FUNDUS", use_column_width=True)
    st.image("/Users/hemasurya/Desktop/RETINAL_PR/assets/image.png", caption="PROLIFERATIVE DIABETIC RETINOPATHY", use_column_width=True)
run()