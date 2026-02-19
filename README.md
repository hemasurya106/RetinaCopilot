# Retinal Disease Grading & Reporting System

## 🩺 Overview
This project is an AI-powered web application for **automated diabetic retinopathy (DR) grading** and **comprehensive clinical report generation** from retinal fundus images. It combines a deep learning model (ResNet) for image classification with Google Gemini for natural language report synthesis, providing explainable, actionable results for clinicians and patients.

---

## 🚀 Features
- **Automated DR Grading:** Upload a retinal fundus image and get an instant DR grade (No DR, Mild, Moderate, Severe, Proliferative DR).
- **Clinical Data Integration:** Enter patient demographics, vitals, and lab results for holistic analysis.
- **Explainable AI:** Visualize model attention with Grad-CAM heatmaps.
- **AI-Generated Medical Reports:** Gemini generates a detailed, professional report combining image findings and clinical data.
- **Downloadable Reports:** Save reports for EMR, referrals, or patient education (as text files).
- **User-Friendly Web Interface:** Built with Streamlit for easy use by clinicians, technicians, and patients.

---

## 🏗️ How It Works
1. **Image Upload:** User uploads a retinal fundus image.
2. **DR Classification:**
   - The image is processed by a fine-tuned ResNet-50 model.
   - The model predicts the DR grade.
3. **Grad-CAM Visualization:**
   - The app generates a heatmap showing which parts of the image influenced the model's decision.
4. **Clinical Data Entry:**
   - User enters age, diabetes type, blood sugar, symptoms, etc.
5. **Report Generation:**
   - Gemini synthesizes the image findings and clinical data into a comprehensive medical report.
6. **Download & Review:**
   - The report can be reviewed on-screen and downloaded as a text file.

---

## 🛠️ Setup Instructions

### 1. **Clone the Repository**
```bash
git clone <your-repo-url>
cd RETINAL_PR
```

### 2. **Install Dependencies**
It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

### 3. **Model Weights**
- Ensure `model_weights-2.pt` (your trained ResNet weights) is present in the project root.

### 4. **Google Gemini API Key**
- Create a `.env` file in the project root:
  ```
  GEMINI_API_KEY=your_gemini_api_key_here
  ```
- You can get a Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 5. **Run the App**
```bash
streamlit run pages/project.py
```
- The app will show a local and a network URL. Open either in your browser.

---

## 💡 Usage Instructions
1. **Upload a retinal fundus image (JPG/PNG).**
2. **Review the predicted DR grade and (optionally) the Grad-CAM heatmap.**
3. **Enter patient clinical data in the form.**
4. **Click "Generate the report" to get a comprehensive AI-generated medical report.**
5. **Download the report as a text file if needed.**

---

## 📝 Example Workflow
1. Upload `sample_fundus.jpg`.
2. See prediction: `Moderate DR`.
3. Enter:
   - Age: 55
   - Diabetes Type: Type 2
   - HbA1c: 8.2
   - ... (other fields)
4. Click **Generate the report**.
5. Review and download the detailed report.

---

## 🧑‍💻 Technologies Used
- **Python 3.10+**
- **Streamlit** (web UI)
- **PyTorch** (deep learning, ResNet)
- **Torchvision** (image transforms)
- **OpenCV, NumPy, Matplotlib** (image processing, Grad-CAM)
- **Google Generative AI (Gemini)** (report generation)
- **python-dotenv** (API key management)

---

## 🩻 Troubleshooting
- **Missing model weights:** Ensure `model_weights-2.pt` is in the project root.
- **Gemini API errors:** Check your `.env` file and internet connection.
- **File upload issues:** Use supported image formats (JPG, PNG).
- **Grad-CAM not showing:** Make sure the toggle is enabled and a valid image is uploaded.
- **Network URL not working:** Ensure your device is on the same WiFi/network as the host computer.

---

## ⚠️ Data & Model Usage Disclaimer

- The deep learning model in this project was trained on data provided under the [Kaggle Competition Data Rules](https://www.kaggle.com/competitions/aptos2019-blindness-detection/rules#7-competition-data).
- The competition data is **not included** in this repository and cannot be redistributed.
- This project and all associated code and models are for **non-commercial, academic, and research purposes only**.
- Do not use this project or any derived models for commercial applications.
- If you have not agreed to the competition rules, you may not use the competition data or any models trained on it.

---

## 📬 Contact
- **Author:** Hemasurya
- **Email:** hemasurya469@gmail.com
- **GitHub:** <your-github-url>

For questions, suggestions, or collaborations, please open an issue or contact us!

---

**Note:**  
- As of the latest update, reports are downloadable as text files only (PDF download has been removed for simplicity and compatibility).
- For best results, use clear, high-quality fundus images and provide as much clinical data as possible. 
