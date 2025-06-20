import streamlit as st
import PyPDF2
import re
import nltk
import pickle
import speech_recognition as sr
import tempfile
import requests
import os

# ---------------------
# Initialization
# ---------------------
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load Models
try:
    tfidf = pickle.load(open("../models/tfidf.pkl", "rb"))
    clf = pickle.load(open("../models/clf.pkl", "rb"))
except FileNotFoundError:
    st.error("❌ tfidf.pkl or clf.pkl not found in /models folder.")
    st.stop()

# Resume Categories
category_mapping = {
    15: "Java Developer", 23: "Testing", 8: "DevOps Engineer", 20: "Python Developer",
    24: "Web Designing", 12: "HR", 13: "Hadoop", 3: "Blockchain", 10: "ETL Developer",
    18: "Operations Manager", 6: "Data Science", 22: "Sales", 16: "Mechanical Engineer",
    1: "Arts", 7: "Database", 11: "Electrical Engineering", 14: "Health and fitness",
    19: "PMO", 4: "Business Analyst", 9: "DotNet Developer", 2: "Automation Testing",
    17: "Network Security Engineer", 21: "SAP Developer", 5: "Civil Engineer", 0: "Advocate"
}

# Backend URL
BACKEND_URL = "http://localhost:5000"

# ---------------------
# Utility Functions
# ---------------------

def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    return "".join([page.extract_text() for page in reader.pages])

def clean_resume(text):
    text = re.sub('http\S+\s*', ' ', text)
    text = re.sub('RT|cc', ' ', text)
    text = re.sub('#\S+', '', text)
    text = re.sub('@\S+', ' ', text)
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]', r' ', text)
    text = re.sub('\s+', ' ', text)
    return text.strip()

def listen_to_user():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("🎙️ Listening... please speak now.")
            audio = recognizer.listen(source, timeout=10)
        text = recognizer.recognize_google(audio)
        st.success(f"🗣️ You said: {text}")
        return text
    except Exception as e:
        st.error(f"Speech Recognition error: {str(e)}")
        return None

def chat_with_backend(user_message):
    try:
        response = requests.post(
            f"{BACKEND_URL}/chat",
            json={"message": user_message},
            timeout=20
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("response", ""), data.get("audio_url", None)
        else:
            st.error(f"❌ Backend error: {response.text}")
            return "", None
    except Exception as e:
        st.error(f"❌ Connection error: {str(e)}")
        return "", None

# ---------------------
# Streamlit App UI
# ---------------------

def main():
    st.set_page_config(
        page_title="AI Resume + Voice Interview Assistant",
        layout="centered"
    )
    st.title("📄 AI Resume + Voice Interview Assistant")

    if 'resume_text' not in st.session_state:
        st.session_state.resume_text = ""

    with st.expander("📁 Upload Resume", expanded=True):
        uploaded_file = st.file_uploader("Upload Resume (PDF or TXT)", type=["pdf", "txt"])
        if uploaded_file:
            st.success("✅ Resume uploaded!")
            if uploaded_file.type == "application/pdf":
                st.session_state.resume_text = extract_text_from_pdf(uploaded_file)
            else:
                st.session_state.resume_text = uploaded_file.read().decode("utf-8")
            st.session_state.resume_text = clean_resume(st.session_state.resume_text)

            with st.expander("📄 View Extracted Resume Text"):
                st.text_area("", value=st.session_state.resume_text, height=300)

            cleaned = tfidf.transform([st.session_state.resume_text])
            prediction = clf.predict(cleaned)[0]
            category = category_mapping.get(prediction, "Unknown")
            st.success(f"🔍 Predicted Job Category: **{category}**")

    st.markdown("---")
    st.subheader("🎤 Voice Interview")

    if st.button("🎤 Start Voice Interview"):
        if not st.session_state.resume_text:
            st.warning("⚠️ Please upload a resume first!")
        else:
            user_question = listen_to_user()
            if user_question:
                with st.spinner("🤖 Generating response..."):
                    full_prompt = (
                        f"I am conducting an interview. Here is the candidate's resume:\n"
                        f"{st.session_state.resume_text}\n\n"
                        f"Interviewer question: {user_question}"
                    )
                    bot_reply, audio_url = chat_with_backend(full_prompt)
                    st.markdown(f"**🤖 Interviewer:** {bot_reply}")
                    if audio_url:
                        st.audio(f"{BACKEND_URL}{audio_url}", format="audio/mp3")

if __name__ == "__main__":
    main()
