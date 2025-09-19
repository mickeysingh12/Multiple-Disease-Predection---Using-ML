# 🧠 Multiple Disease Prediction Using ML

A Streamlit-based web application that predicts the likelihood of **Diabetes**, **Heart Disease**, and **Parkinson's Disease** using pre-trained machine learning models.

## 🚀 Features
- 🔍 Predicts three major diseases based on user input
- 💾 Uses saved `.sav` models for fast and accurate inference
- 🖥️ Interactive UI built with Streamlit
- ⚠️ Gracefully handles missing model files with warnings

## 📦 Requirements
- Python 3.8 or higher (tested with CPython 3.11/3.13)
- Install dependencies:
  ```bash
  python -m pip install -r requirements.txt

▶️ How to Run
1 Clone the repository:
  git clone https://github.com/your-username/multiple-disease-prediction-ml.git
  cd multiple-disease-prediction-ml

2 Ensure the following model files are placed inside the saved models/ directory:
  - diabetes_model.sav
  - heart_disease_model.sav
  - parkinsons_model.sav

3 Launch the app:
streamlit run "multiple disease pred.py" 
                OR
& 'c:\Python313\python.exe' -m streamlit run "D:\project\Projects\Multiple Disease Prediction System-20250919T080746Z-1-001\
Multiple Disease Prediction System\multiple disease pred.py"

📁 Project Structure
├── multiple disease pred.py
├── requirements.txt
├── saved models/
│   ├── diabetes_model.sav
│   ├── heart_disease_model.sav
│   └── parkinsons_model.sav

🧪 Model & Dataset Info
- Models trained using publicly available datasets:
  - Diabetes: PIMA Indian Diabetes Dataset
  - Heart Disease: Cleveland Heart Disease Dataset
  - Parkinson’s: UCI Parkinson’s Dataset
- Preprocessing includes scaling, feature selection, and model tuning
- Models saved using joblib or pickle as .sav files

☁️ Deployment Options
You can deploy this app on:
- Streamlit Cloud
- Hugging Face Spaces
- Render
- Heroku

🛠 Troubleshooting
- Missing model files: App will run but show a warning.
- Streamlit not found: Ensure you're using the correct Python environment:
pip show streamlit

🙌 Author
Developed by Abhishek Singh
📍 Mumbai, India
💼 Data Analyst | Aspiring Data Engineer & ML Enthusiast

💬 Feedback & Contributions
Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

Let me know if you'd like to add:
- 📸 Screenshots or GIFs of the app in action
- 🧪 Model training details or dataset sources
- ☁️ Deployment instructions (e.g., Streamlit Cloud or Hugging Face Spaces)
- 🏷️ Badges like Python version, license, or last updated

I can also help you write a crisp project summary for your portfolio site or LinkedIn.


