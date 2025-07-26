import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torch.nn import functional as F
from datetime import datetime
import google.generativeai as genai
from PIL import Image 
from dotenv import load_dotenv
import os
load_dotenv(override=True)
from io import BytesIO
from fpdf import FPDF
# Load trained model
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model.load_state_dict(torch.load("/Users/hemasurya/Desktop/RETINAL_PR/model_weights-2.pt", map_location=torch.device('cpu')))
    model.eval()
    return model

def format_clinical_summary(clinical_data):
    """Format clinical data for display"""
    summary = f"""
Age: {clinical_data['age']} | Sex: {clinical_data['sex']} | Diabetes: {clinical_data['diabetes']}
HbA1c: {clinical_data['hba1c']}% | FBS: {clinical_data['fasting_blood_sugar']} mg/dL | BP: {clinical_data['blood_pressure']}
Sugar Control: {clinical_data['sugar_control_status']} | Duration: {clinical_data['diabetes_duration']} years
Symptoms: Blurry Vision({clinical_data['blurry_vision']}), Eye Pain({clinical_data['eye_pain']}), Floaters({clinical_data['floaters_or_flashes']})
Prior DR: {clinical_data['prior_dr_diagnosis']} | Prior Laser: {clinical_data['prior_laser_treatment']}
    """
    return summary.strip()

def generate_gradcam(model, input_tensor, target_class):
    """Generate Grad-CAM heatmap"""
    # Hook the feature map and gradients
    features = []
    gradients = []
    def forward_hook(module, input, output):
        features.append(output)
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])
    # Register hooks to the last conv layer
    handle_fwd = model.layer4[-1].register_forward_hook(forward_hook)
    handle_bwd = model.layer4[-1].register_backward_hook(backward_hook)
    # Forward (with grad enabled)
    output = model(input_tensor)
    model.zero_grad()
    class_loss = output[0, target_class]
    class_loss.backward()
    # Get hooked data
    grads_val = gradients[0].cpu().data.numpy()[0]
    fmap = features[0].cpu().data.numpy()[0]
    # Global average pooling of gradients
    weights = np.mean(grads_val, axis=(1, 2))
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * fmap[i, :, :]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8)
    # Remove hooks
    handle_fwd.remove()
    handle_bwd.remove()
    return cam

def format_report_filename(age, sex, diabetes):
    """Format filename for report download"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"DR_Report_{age}_{sex}_{diabetes}_{timestamp}.txt"

def clean_text_for_pdf(text):
    # Replace en dash and em dash with hyphen
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    # Replace smart quotes with straight quotes
    text = text.replace('\u201c', '"').replace('\u201d', '"').replace('\u2018', "'").replace('\u2019', "'")
    # Add more replacements as needed
    return text

def text_to_pdf(report_text, title="Medical Report"):
    """Convert plain text report to PDF and return as BytesIO object."""
    report_text = clean_text_for_pdf(report_text)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, title, ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    # Split text into lines and add
    for line in report_text.split('\n'):
        pdf.multi_cell(0, 10, line)
    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    return pdf_output

model = load_model()
st.title("📥 Retinal Disease Grading")

# --- Robust file uploader persistence ---
if 'uploaded_file' not in st.session_state:
    st.session_state['uploaded_file'] = None

uploaded_file = st.file_uploader("Upload a Retinal Fundus Image (JPG/PNG)", type=["jpg", "jpeg", "png"], key="uploaded_file_widget")
if uploaded_file is not None:
    st.session_state['uploaded_file'] = uploaded_file

# --- Always show input widgets if file is uploaded and DR grade >= 1 ---
def show_clinical_inputs():
    if 'clinical_data' not in st.session_state:
        st.session_state['clinical_data'] = {}
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, key="age")
        sex = st.selectbox("What is your sex?", ("Male", "Female", "Others", "Dont Want to specify"), key="sex")
        diabetes = st.selectbox("Type of Diabetes", ["Type 1", "Type 2", "Gestational", "Other","None"], key="diabetes")
        sugar_control_status = st.radio("Blood Sugar Control", ["Good", "Moderate", "Poor"], key="sugar_control_status")
        pulse_rate = st.number_input("Pulse Rate (bpm)", min_value=30, max_value=200, key="pulse_rate")
        blurry_vision = st.radio("Blurry Vision?", ["Yes", "No"], key="blurry_vision")
        eye_pain = st.radio("Eye Pain?", ["Yes", "No"], key="eye_pain")
        prior_laser_treatment = st.radio("Prior Laser Treatment?", ["Yes", "No"], key="prior_laser_treatment")
    with col2:
        hba1c = st.number_input("HbA1c (%)", min_value=0.0, max_value=20.0, step=0.1, key="hba1c")
        fasting_blood_sugar = st.number_input("Fasting Blood Sugar (mg/dL)", min_value=40, max_value=500, key="fasting_blood_sugar")
        diabetes_duration = st.number_input("Years with Diabetes", min_value=0, max_value=80, key="diabetes_duration")
        blood_pressure = st.text_input("Blood Pressure (e.g., 140/90)", key="blood_pressure")
        floaters_or_flashes = st.radio("Floaters or Flashes?", ["Yes", "No"], key="floaters_or_flashes")
        prior_dr_diagnosis = st.radio("Previously Diagnosed with DR?", ["Yes", "No"], key="prior_dr_diagnosis")
        clinical_notes = st.text_area("Additional Clinical Notes(if none specify NA)", key="clinical_notes")
    st.session_state['clinical_data'] = {
        'age': age,
        'sex': sex,
        'diabetes': diabetes,
        'sugar_control_status': sugar_control_status,
        'pulse_rate': pulse_rate,
        'blurry_vision': blurry_vision,
        'eye_pain': eye_pain,
        'prior_laser_treatment': prior_laser_treatment,
        'hba1c': hba1c,
        'fasting_blood_sugar': fasting_blood_sugar,
        'diabetes_duration': diabetes_duration,
        'blood_pressure': blood_pressure,
        'floaters_or_flashes': floaters_or_flashes,
        'prior_dr_diagnosis': prior_dr_diagnosis,
        'clinical_notes': clinical_notes
    }

# --- Button flag for report generation ---
if 'generate_report' not in st.session_state:
    st.session_state['generate_report'] = False

def run():
    uploaded_file = st.session_state.get('uploaded_file')
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.markdown("Check for your output below")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

        input_tensor = preprocess(image).unsqueeze(0)

        # Normal prediction (no gradients needed)
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            grades = ['No DR', 'Mild/No DR', 'Moderate', 'Severe', 'Proliferative DR']
            st.success(f"🩺 **Predicted Grade:** {grades[predicted.item()]}")
        # Grad-CAM (requires gradients, so input_tensor must require grad)
        show_gradcam = st.checkbox("Show Grad-CAM Heatmap", value=False)
        if show_gradcam:
            input_tensor_for_gradcam = input_tensor.clone().detach()
            input_tensor_for_gradcam.requires_grad = True
            cam = generate_gradcam(model, input_tensor_for_gradcam, predicted.item())
            # Prepare overlay
            img_np = np.array(image.resize((224, 224)))
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = np.float32(heatmap) / 255 + np.float32(img_np) / 255
            overlay = overlay / np.max(overlay)
            # Show Grad-CAM
            st.markdown("###Grad-CAM Heatmap (Model Attention)")
            st.image((overlay * 255).astype(np.uint8), caption="Grad-CAM Heatmap", use_column_width=True)
        
        if(predicted.item()>=2):
            st.header("Enter Clinical Data for further Diagnosis")
            show_clinical_inputs()
            # --- Button to trigger report generation ---
            if st.button("Generate the report", type="primary"):
                st.session_state['generate_report'] = True
            if st.session_state['generate_report']:
                clinical_data = st.session_state['clinical_data']
                # Configure Gemini API
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                model_1 = genai.GenerativeModel(os.getenv("MODEL"))
                prompt_text = f"""
                    Act as an expert ophthalmologist analyzing a retinal fundus image.
                    **Provided Information:**
                    The image is predicted to be at the {grades[predicted.item()]} severity level of diabetic retinopathy.

                    **Your Task:**
                    Based on the visual evidence in the image, provide a detailed medical report. Your analysis should be structured as follows:

                    1.  **Optic Disc and Macula:** Describe the appearance, color, contours, and cup-to-disc ratio of the optic disc. Comment on the state of the macula and foveal reflex.
                    2.  **Retinal Vasculature:** Assess the arteries and veins, noting any tortuosity, caliber changes, or AV nicking.
                    3.  **Key Pathological Findings:** Systematically identify and describe any visible signs of retinopathy, such as:
                        * Microaneurysms
                        * Hemorrhages (dot, blot, flame-shaped)
                        * Exudates (hard or soft/cotton-wool spots)
                        * Neovascularization
                        * Any other abnormalities.
                    4.  **Impression:** Provide a summary impression that correlates the findings with the provided grade of '{grades[predicted.item()]}'.
                    """         
                prompt = [prompt_text, image]
                response_1 = model_1.generate_content(prompt)   
                prompt_text_2=f"""
                Act as an expert medical report writer. You have received:
                1. A detailed ophthalmological analysis of a retinal fundus image
                2. Clinical patient data
                3. AI-predicted diabetic retinopathy grade
                

                **Ophthalmological Analysis:**
                {response_1.text}
                
                **Patient Clinical Data:**
                - Age: {clinical_data['age']} years
                - Sex: {clinical_data['sex']}
                - Diabetes Type: {clinical_data['diabetes']}
                - Diabetes Duration: {clinical_data['diabetes_duration']} years
                - HbA1c: {clinical_data['hba1c']}%
                - Fasting Blood Sugar: {clinical_data['fasting_blood_sugar']} mg/dL
                - Blood Pressure: {clinical_data['blood_pressure']}
                - Pulse Rate: {clinical_data['pulse_rate']} bpm
                - Sugar Control Status: {clinical_data['sugar_control_status']}
                - Symptoms: Blurry Vision({clinical_data['blurry_vision']}), Eye Pain({clinical_data['eye_pain']}), Floaters({clinical_data['floaters_or_flashes']})
                - Prior DR Diagnosis: {clinical_data['prior_dr_diagnosis']}
                - Prior Laser Treatment: {clinical_data['prior_laser_treatment']}
                - Clinical Notes: {clinical_data['clinical_notes']}

                **AI Predicted Grade:** {grades[predicted.item()]}

                **Your Task:**
                Synthesize the provided clinical data and ophthalmological findings for a patient. Identify key systemic risk factors based on the vitals and correlate them with the retinal findings. Generate a comprehensive, professional medical report using the exact structure below.

                # DIABETIC RETINOPATHY COMPREHENSIVE MEDICAL REPORT
                
                ## EXECUTIVE SUMMARY
                [Brief overview of findings and urgency level]

                ## PATIENT INFORMATION
                [Demographics and diabetes history]

                ## CLINICAL PRESENTATION
                [Current symptoms and vital signs]

                ## OPHTHALMOLOGICAL FINDINGS
                [Detailed analysis from the retinal image]

                ## LABORATORY VALUES
                [Relevant lab results and their interpretation and analyse all the vitals and check for any abnormality and use it to generate the rpeort too]

                ## DIAGNOSIS
                [Confirmed diagnosis with grade]

                ## TREATMENT RECOMMENDATIONS
                [Specific treatment plan based on grade and clinical data]

                ## FOLLOW-UP PLAN
                [Timeline for follow-up appointments]

                ## PROGNOSIS
                [Expected outcome and risk factors]

                ## DISCLAIMER
                [Standard medical disclaimer]

                Make the report professional, comprehensive, and actionable for healthcare providers.
                """
                
                # Generate comprehensive report with progress indicator
                with st.spinner("🤖 AI is generating comprehensive medical report..."):
                    try:
                        # Get the final comprehensive report
                        response_2 = model_1.generate_content(prompt_text_2)
                        # Display clinical summary
                        st.markdown("### 📋 Clinical Data Summary")
                        st.text(format_clinical_summary(clinical_data))
                        # Display the comprehensive report
                        st.markdown("## 📋 Comprehensive Medical Report")
                        st.markdown("---")
                        st.markdown(response_2.text)
                        # Download functionality
                        filename = format_report_filename(clinical_data['age'], clinical_data['sex'], clinical_data['diabetes'])
                        st.download_button(
                            label="📥 Download Report as Text",
                            data=response_2.text,
                            file_name=filename,
                            mime="text/plain"
                        )
                        # PDF download
                        pdf_bytes = text_to_pdf(response_2.text, title="Comprehensive Medical Report")
                        st.download_button(
                            label="📄 Download Report as PDF",
                            data=pdf_bytes,
                            file_name=filename.replace('.txt', '.pdf'),
                            mime="application/pdf"
                        )
                        st.success("✅ Comprehensive medical report generated successfully!")
                    except Exception as e:
                        st.error(f"❌ Error generating report: {str(e)}")
                        st.info("Please check your Gemini API key and internet connection.")
        elif predicted.item() == 1:
            st.markdown("You have mild chances of Diabetic Retinopathy")
            show_clinical_inputs()
            clinical_data = st.session_state.get('clinical_data', {})
            # Button to trigger AI second opinion
            if st.button('AI Second Opinion: Analyze Image and Vitals'):
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                model_llm = genai.GenerativeModel('gemini-1.5-pro')
                # Clean, professional prompt for LLM
                prompt = f"""
                You are an expert ophthalmologist. The automated model predicted 'Mild Diabetic Retinopathy (DR)' for this patient's retinal fundus image.
                
                Please:
                1. Carefully examine the image for any signs of mild DR or other retinal abnormalities.
                2. Analyze all vitals and lab values for supporting evidence of mild DR or related risk factors.
                3. Provide a clear, concise report summarizing:
                   - Any findings in the image
                   - Any abnormal vitals or labs that support or contradict the mild DR diagnosis
                   - The overall health status and recommendations
                
                **Patient Clinical Data:**
                - Age: {clinical_data.get('age', 'N/A')}
                - Sex: {clinical_data.get('sex', 'N/A')}
                - Diabetes Type: {clinical_data.get('diabetes', 'N/A')}
                - Diabetes Duration: {clinical_data.get('diabetes_duration', 'N/A')} years
                - HbA1c: {clinical_data.get('hba1c', 'N/A')}%
                - Fasting Blood Sugar: {clinical_data.get('fasting_blood_sugar', 'N/A')} mg/dL
                - Blood Pressure: {clinical_data.get('blood_pressure', 'N/A')}
                - Pulse Rate: {clinical_data.get('pulse_rate', 'N/A')} bpm
                - Sugar Control Status: {clinical_data.get('sugar_control_status', 'N/A')}
                - Symptoms: Blurry Vision({clinical_data.get('blurry_vision', 'N/A')}), Eye Pain({clinical_data.get('eye_pain', 'N/A')}), Floaters({clinical_data.get('floaters_or_flashes', 'N/A')})
                - Prior DR Diagnosis: {clinical_data.get('prior_dr_diagnosis', 'N/A')}
                - Prior Laser Treatment: {clinical_data.get('prior_laser_treatment', 'N/A')}
                - Clinical Notes: {clinical_data.get('clinical_notes', 'N/A')}
                
                ---
                
                **Your Task:**
                If there is evidence of DR, generate a comprehensive, professional medical report using the structure below. If not, state that there is no evidence of DR and provide any relevant recommendations.
                
                # DIABETIC RETINOPATHY COMPREHENSIVE MEDICAL REPORT
                
                ## EXECUTIVE SUMMARY
                [Brief overview of findings and urgency level]
                
                ## PATIENT INFORMATION
                [Demographics and diabetes history]
                
                ## CLINICAL PRESENTATION
                [Current symptoms and vital signs]
                
                ## OPHTHALMOLOGICAL FINDINGS
                [Detailed analysis from the retinal image]
                
                ## LABORATORY VALUES
                [Relevant lab results, interpretation, and analysis of all vitals. Check for any abnormality and use it to generate the report.]
                
                ## DIAGNOSIS
                [Confirmed diagnosis with grade]
                
                ## TREATMENT RECOMMENDATIONS
                [Specific treatment plan based on grade and clinical data]
                
                ## FOLLOW-UP PLAN
                [Timeline for follow-up appointments]
                
                ## PROGNOSIS
                [Expected outcome and risk factors]
                
                ## DISCLAIMER
                [Standard medical disclaimer]
                
                Make the report professional, comprehensive, and actionable for healthcare providers.
                """
                import io
                img_bytes = io.BytesIO()
                image.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                with st.spinner("🤖 AI is analyzing your image and vitals for a second opinion..."):
                    try:
                        response = model_llm.generate_content([prompt, ("image", img_bytes.getvalue(), "image/png")])
                        st.markdown("## 📋 AI Second Opinion Report")
                        st.markdown("---")
                        st.markdown(response.text)
                        # Download button for the report (text only)
                        filename = format_report_filename(clinical_data.get('age', 'N/A'), clinical_data.get('sex', 'N/A'), clinical_data.get('diabetes', 'N/A'))
                        st.download_button(
                            label="📥 Download Second Opinion Report as Text",
                            data=response.text,
                            file_name=filename,
                            mime="text/plain"
                        )
                    except Exception as e:
                        st.error(f"Could not analyze image and vitals with LLM: {e}")
        else:
            show_clinical_inputs()
            clinical_data = st.session_state.get('clinical_data', {})
            if 'check_vitals' not in st.session_state:
                st.session_state['check_vitals'] = False
            if st.button('Check Vitals for Abnormalities'):
                st.session_state['check_vitals'] = True
            if st.session_state['check_vitals']:
                # Use Gemini LLM to analyze vitals and labs for abnormalities
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                model_llm = genai.GenerativeModel('gemini-1.5-pro')
                prompt = f"""
                You are a medical expert. Analyze the following patient's clinical data for any abnormalities in vitals and lab values. List any abnormal findings and provide a brief summary. If all values are normal, state that all vitals are within normal range.

                Patient Clinical Data:
                - Age: {clinical_data.get('age', 'N/A')}
                - Sex: {clinical_data.get('sex', 'N/A')}
                - Diabetes Type: {clinical_data.get('diabetes', 'N/A')}
                - Diabetes Duration: {clinical_data.get('diabetes_duration', 'N/A')} years
                - HbA1c: {clinical_data.get('hba1c', 'N/A')}%
                - Fasting Blood Sugar: {clinical_data.get('fasting_blood_sugar', 'N/A')} mg/dL
                - Blood Pressure: {clinical_data.get('blood_pressure', 'N/A')}
                - Pulse Rate: {clinical_data.get('pulse_rate', 'N/A')} bpm
                - Sugar Control Status: {clinical_data.get('sugar_control_status', 'N/A')}
                - Symptoms: Blurry Vision({clinical_data.get('blurry_vision', 'N/A')}), Eye Pain({clinical_data.get('eye_pain', 'N/A')}), Floaters({clinical_data.get('floaters_or_flashes', 'N/A')})
                - Prior DR Diagnosis: {clinical_data.get('prior_dr_diagnosis', 'N/A')}
                - Prior Laser Treatment: {clinical_data.get('prior_laser_treatment', 'N/A')}
                - Clinical Notes: {clinical_data.get('clinical_notes', 'N/A')}
                """
                with st.spinner("Analyzing your vitals and labs with AI..."):
                    try:
                        response = model_llm.generate_content(prompt)
                        st.info(response.text)
                    except Exception as e:
                        st.error(f"Could not analyze vitals with LLM: {e}")

        

if __name__ == "__main__" or True:
    run()
st.markdown("""
    *Note: This prediction is made for educational demonstration purposes only, based on academic dataset regulations.*
    """)