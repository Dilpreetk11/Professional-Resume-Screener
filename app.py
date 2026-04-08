import streamlit as st
import pickle
import docx
import PyPDF2
import re
import time

# ==============================================================================
# 1. UI Configuration & Clean Light Mode CSS
# ==============================================================================
st.set_page_config(page_title="AI Resume Screening", page_icon="📄", layout="wide")

page_bg_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Styles */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #111827;
    background-color: #f9fafb;
}

/* Base App Background */
.stApp {
    background-color: #f9fafb;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e5e7eb;
}

/* Text & Headers */
h1, h2, h3, h4, h5, h6 {
    color: #111827;
    font-weight: 700;
}

p, span, div {
    color: #374151;
}

/* Upload Box Dropzone */
.stFileUploadDropzone {
    background-color: #ffffff;
    border: 1px dashed #d1d5db;
    border-radius: 8px;
    padding: 20px;
}
.stFileUploadDropzone:hover {
    border-color: #2563eb;
    background-color: #eff6ff;
}

/* Metric Display Cards */
.metric-container {
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 24px;
    height: 100%;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
}

/* Micro-Titles */
.metric-title {
    font-size: 0.875rem;
    font-weight: 600;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
}

/* Value text */
.metric-value {
    font-size: 1.875rem;
    font-weight: 700;
    color: #111827;
    margin-bottom: 0;
}

/* Skill Tags */
.skill-tag {
    display: inline-block;
    padding: 4px 12px;
    background-color: #f3f4f6;
    color: #374151;
    border-radius: 4px;
    font-size: 0.875rem;
    font-weight: 500;
    margin: 4px;
    border: 1px solid #e5e7eb;
}

/* Primary Button */
.stButton > button {
    background-color: #2563eb;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 500;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
}
.stButton > button:hover {
    background-color: #1d4ed8;
    color: white;
}

/* Fix sidebar text color explicitly */
[data-testid="stSidebar"] p, [data-testid="stSidebar"] div, [data-testid="stSidebar"] span {
    color: #374151 !important;
}

/* Revert text selection visibility */
::selection {
    background: #bfdbfe;
    color: #1e3a8a;
}
</style>
"""
st.markdown(page_bg_css, unsafe_allow_html=True)

# ==============================================================================
# 2. Machine Learning & NLP Backend
# ==============================================================================
@st.cache_resource
def load_models():
    try:
        clf = pickle.load(open('clf.pkl', 'rb'))
        tfidf = pickle.load(open('tfidf.pkl', 'rb'))
        le = pickle.load(open('encoder.pkl', 'rb'))
        return clf, tfidf, le
    except FileNotFoundError:
        return None, None, None

clf, tfidf, le = load_models()

# Comprehensive Skills Dictionary for Extraction
SKILLS_DB = [
    'Python', 'Machine Learning', 'Data Science', 'Data Analysis', 'SQL', 'Java', 'C++', 'C#', 
    'JavaScript', 'TypeScript', 'React', 'Angular', 'Vue.js', 'Node.js', 'Express', 'Django', 
    'Flask', 'Spring Boot', 'AWS', 'Azure', 'Google Cloud', 'Docker', 'Kubernetes', 'Jenkins', 
    'CI/CD', 'Git', 'Agile', 'Scrum', 'Leadership', 'Communication', 'Project Management', 
    'Problem Solving', 'Teamwork', 'Critical Thinking', 'Business Analysis', 'Marketing', 
    'SEO', 'Sales', 'Customer Service', 'Ruby', 'PHP', 'HTML', 'CSS', 'PostgreSQL', 'MongoDB',
    'Redis', 'GraphQL', 'REST API', 'Cybersecurity', 'Network Security', 'Linux', 'Bash', 
    'TensorFlow', 'PyTorch', 'Keras', 'NLP', 'Computer Vision', 'Tableau', 'Power BI', 'Excel',
    'Financial Analysis', 'Accounting', 'HR', 'Recruiting', 'Public Speaking', 'UI/UX Design',
    'Figma', 'Adobe Creative Suite', 'AutoCAD', 'MATLAB', 'R', 'Go', 'Rust', 'Swift', 'Kotlin',
    'Android Development', 'iOS Development', 'Solidity', 'Blockchain', 'Big Data', 'Hadoop',
    'Spark', 'Kafka', 'ETL', 'Salesforce', 'SAP', 'ERP', 'Supply Chain Management'
]

def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', '  ', cleanText)
    cleanText = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText

def extract_text(file):
    file_ext = file.name.split('.')[-1].lower()
    text = ""
    if file_ext == 'pdf':
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    elif file_ext == 'docx':
        doc = docx.Document(file)
        for p in doc.paragraphs:
            text += p.text + "\n"
    elif file_ext == 'txt':
        try:
            text = file.read().decode('utf-8')
        except UnicodeDecodeError:
            text = file.read().decode('latin-1')
    return text

def predict_category(text, clf, tfidf, le):
    cleanedText = cleanResume(text)
    vectorizedText = tfidf.transform([cleanedText])
    try:
        prediction = clf.predict(vectorizedText)
    except:
        prediction = clf.predict(vectorizedText.toarray())
    return le.inverse_transform(prediction)[0]

def extract_skills(text):
    found_skills = []
    text_lower = text.lower()
    for skill in SKILLS_DB:
        # Word boundary ensures we don't match 'R' inside 'HR' or 'Java' inside 'Javascript'
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.append(skill)
    return list(set(found_skills))

# ==============================================================================
# 3. Main Streamlit Layout
# ==============================================================================
col1, col2 = st.columns([0.6, 4])
with col1:
    # A cleaner, more professional icon
    st.image("https://cdn-icons-png.flaticon.com/512/3616/3616234.png", width=65) 
with col2:
    st.title("Professional Resume Screener")
    st.markdown("Automated Talent Extraction & Profession Prediction")

st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload Profile Data", type=["pdf", "docx", "txt"])
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.info("System Engine: Active\n\nParser: TF-IDF + SVC\n\nSkill Extractor: NLP Matcher")

# Main Panel
if clf is None:
    st.error("🚨 Models not found! Please run the training script.")
else:
    if uploaded_file is None:
        st.write("👈 Upload a candidate resume file from the sidebar to begin processing.")
    else:
        with st.spinner("Analyzing document metrics..."):
            time.sleep(0.6) # subtle animation delay for UX
            resume_text = extract_text(uploaded_file)
            
            if not resume_text.strip():
                st.warning("Could not extract any text from the file.")
            else:
                # Run the AI Pipelines
                category = predict_category(resume_text, clf, tfidf, le)
                skills = extract_skills(resume_text)
                
                # Results Dashboard
                st.subheader("Candidate Overview")
                st.markdown("<br>", unsafe_allow_html=True)
                
                dashboard_col1, dashboard_col2 = st.columns([1, 1])
                
                with dashboard_col1:
                    st.markdown(
                        f"""
                        <div class="metric-container">
                            <div class="metric-title">Predicted Profession</div>
                            <div class="metric-value">{category}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                with dashboard_col2:
                    skills_html = ""
                    if skills:
                        # Limit to top 15 skills for cleaner UI look
                        displayed_skills = skills[:15] 
                        for s in displayed_skills:
                            skills_html += f'<span class="skill-tag">{s}</span>'
                        if len(skills) > 15:
                            skills_html += f'<span class="skill-tag" style="background:#ffffff;">+{len(skills)-15} more</span>'
                    else:
                        skills_html = "<span style='color:#6b7280; font-size: 0.9rem;'>No verified skills matched from database.</span>"
                        
                    st.markdown(
                        f"""
                        <div class="metric-container">
                            <div class="metric-title">Verified Top Skills</div>
                            <div style="margin-top: 10px;">{skills_html}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                st.markdown("<br><br>", unsafe_allow_html=True)
                with st.expander("Show Raw Parsed Resume File"):
                    st.text_area("", resume_text, height=300)
