# 🏢 Professional Resume Screener (ATS)

An advanced, automated Talent Intelligence and Applicant Tracking System (ATS) pipeline built completely with Python and Streamlit. This application allows recruiters to rapidly screen candidate resumes by using Machine Learning to accurately predict the candidate's professional job category and utilizing Natural Language Processing (NLP) to extract their core skills.

## ✨ Key Features

- **Automated Profession Prediction:** Employs a robust `scikit-learn` Machine Learning pipeline using TF-IDF (Term Frequency-Inverse Document Frequency) text vectorization combined with an SVC (Support Vector Classifier) / One-vs-Rest strategy to categorize resumes based on historical data.
- **Top Skill Extraction:** Implements an intelligent NLP boundary-matching algorithm against a comprehensive database of 100+ highly sought-after industry skills (e.g., Python, React, Leadership, Project Management) to map exactly what the candidate knows.
- **Multi-Format Document Parsing:** Ingests raw text seamlessly from `.PDF`, `.DOCX`, and standard `.TXT` files using `PyPDF2` and `python-docx`.
- **Premium SaaS UI:** Features an incredibly clean, minimalistic "Corporate Light" dashboard layout modeled after top-tier Silicon Valley HR software. 
- **Lightning Fast:** Predictions happen locally within a fraction of a second, drastically streamlining the recruitment verification step.

## 🛠️ Technology Stack

- **Frontend Application Layer:** [Streamlit](https://streamlit.io/)
- **Machine Learning Layer:** `scikit-learn`, `numpy`, `pandas`
- **Document Processors:** `PyPDF2` (PDFs), `python-docx` (Word documents)
- **Object Serialization:** `joblib` & `pickle` (Caching `clf.pkl` & `tfidf.pkl`)

## ⚙️ Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/Dilpreetk11/Professional-Resume-Screener.git
cd Professional-Resume-Screener
```

### 2. Install dependencies
Ensure you are using **Python 3.10+**.
```bash
pip install -r requirements.txt
```

### 3. (Optional) Re-Train the Model
If you make changes to the `.csv` dataset and wish to generate new `.pkl` categorization models, you can run the independent training pipeline at any time:
```bash
python train_model.py
```

### 4. Run the Streamlit Application
Start local deployment. The system will create a local web portal.
```bash
python -m streamlit run app.py
```

## 🧠 Model Architecture details
The fundamental backend algorithm works via an NLP cleaning pipeline. Resumes have severe noise (excessive linebreaks, URLs, twitter handles, and odd unicode chars) which are structurally sanitized using regex boundaries. The pristine output is transformed into dimensional arrays using a `TfidfVectorizer` yielding numerical representations of skill-weightage. A `LabelEncoder` translates string job classifications into numeric vectors for training the cross-categorical class map.
