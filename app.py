import streamlit as st
import fitz  # PyMuPDF
import spacy
import pandas as pd

# Load NLP model
nlp = spacy.load("en_core_web_md")

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

st.title("AI Resume Screener & Ranker")

# Input: Job Description
jd_text = st.text_area("Paste the Job Description here:", height=200)

# Input: Multiple Resumes
uploaded_files = st.file_uploader("Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)

if st.button("Score Resumes") and jd_text and uploaded_files:
    results = []
    jd_doc = nlp(jd_text.lower())

    for file in uploaded_files:
        resume_text = extract_text_from_pdf(file)
        resume_doc = nlp(resume_text.lower())
        
        # Calculate Similarity
        score = resume_doc.similarity(jd_doc)
        
        results.append({
            "Candidate Name": file.name,
            "Match Score (%)": round(score * 100, 2)
        })

    # Display Results in a Table
    df = pd.DataFrame(results).sort_values(by="Match Score (%)", ascending=False)
    st.table(df)
