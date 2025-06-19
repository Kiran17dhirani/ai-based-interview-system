import streamlit as st
import PyPDF2
import re
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from elevenlabs import generate, play, set_api_key
import openai
import speech_recognition as sr
import tempfile
import requests

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Set API Keys (replace with your actual keys)
ELEVENLABS_API_KEY = "your-elevenlabs-api-key"
OPENAI_API_KEY = "your-openai-api-key"
set_api_key(ELEVENLABS_API_KEY)
openai.api_key = OPENAI_API_KEY

# Load pre-trained models
tfidf = pickle.load(open("tfidf.pkl", "rb"))
clf = pickle.load(open("clf.pkl", "rb"))

# Resume Category Mapping
category_mapping = {
    15: "Java Developer", 23: "Testing", 8: "DevOps Engineer", 20: "Python Developer",
    24: "Web Designing", 12: "HR", 13: "Hadoop", 3: "Blockchain", 10: "ETL Developer",
    18: "Operations Manager", 6: "Data Science", 22: "Sales", 16: "Mechanical Engineer",
    1: "Arts", 7: "Database", 11: "Electrical Engineering", 14: "Health and fitness",
    19: "PMO", 4: "Business Analyst", 9: "DotNet Developer", 2: "Automation Testing",
    17: "Network Security Engineer", 21: "SAP Developer", 5: "Civil Engineer", 0: "Advocate"
}

# ---------------------
# Utility Functions
# ---------------------
def clean_resume(text):
    text = re.sub('http\S+\s*', ' ', text)
    text = re.sub('RT|cc', ' ', text)
    text = re.sub('#\S+', '', text)
    text = re.sub('@\S+', ' ', text)
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]', r' ', text)
    text = re.sub('\s+', ' ', text)
    return text.strip()

def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def listen_to_user():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        st.info("🎙️ Listening... please speak now.")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        st.success(f"🗣️ You said: {text}")
        return text
    except sr.UnknownValueError:
        st.error("Sorry, I could not understand the audio.")
        return None
    except sr.RequestError:
        st.error("Speech Recognition service error.")
        return None

def speak_response(text):
    audio = generate(text=text, voice="Rachel", model="eleven_monolingual_v1")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        f.write(audio)
        return f.name

def get_chatgpt_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# ---------------------
# Streamlit Interface
# ---------------------
def main():
    st.set_page_config(page_title="AI ATS + Voice Interview Bot", layout="centered")
    st.title("📄 AI Resume + Voice Interview Assistant")

    resume_text = ""
    uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "txt"])

    if uploaded_file is not None:
        st.success("Resume uploaded successfully!")
        if uploaded_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(uploaded_file)
        else:
            resume_text = uploaded_file.read().decode("utf-8")

        resume_text = clean_resume(resume_text)
        st.text_area("📄 Extracted Resume", value=resume_text, height=300)

        cleaned = tfidf.transform([resume_text])
        prediction = clf.predict(cleaned)[0]
        category = category_mapping.get(prediction, "Unknown")
        st.success(f"🔍 Predicted Category: {category}")

    st.markdown("---")
    st.subheader("🎤 AI Interview Chat")

    if st.button("Start Voice Interview"):
        user_question = listen_to_user()
        if user_question and resume_text:
            full_prompt = f"I am an interviewer. Here is the candidate's resume:\n{resume_text}\n\nNow answer this question: {user_question}"
            bot_reply = get_chatgpt_response(full_prompt)
            st.markdown(f"**🤖 Bot:** {bot_reply}")
            audio_path = speak_response(bot_reply)
            st.audio(audio_path, format="audio/mp3", autoplay=True)

   API_BASE_URL = "https://your-api-domain.com"

   def get_next_ai_question():
       response = requests.get(f"{API_BASE_URL}/next-question")
      if response.status_code == 200:
         return response.json().get("question")
      return None

   def submit_candidate_answer(answer_text):
     payload = {"answer": answer_text}
     response = requests.post(f"{API_BASE_URL}/analyze-answer", json=payload)
     if response.status_code == 200:
         return response.json()
      return None


if __name__ == "__main__":
    main()



