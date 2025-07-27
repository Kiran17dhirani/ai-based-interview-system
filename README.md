## AI-Based Interview System
An intelligent AI-powered interview system that analyzes resumes, provides feedback, and simulates interview responses. This project uses Machine Learning, NLP, and Streamlit to streamline
the candidate screening process with interactive resume parsing and smart assistance.

## 🚀 Features

- 📄 Resume Parsing and Extraction using NLP
- 🎯 ML Classification Model to assess candidate profiles
- 🗣️ Voice-based Interview Assistance with feedback
- 🌐 Streamlit Web App for interactive UI
- 📊 Visual output and result interpretation
- 🐳 Dockerized for easy deployment


## 📁 Project Structure

interview-system-with-ai-ml/
├── .streamlit/                 # Streamlit config files
├── templates/                  # HTML templates for additional pages
├── app.py                      # Main entry point for Streamlit app
├── backend_app.py              # Handles logic behind the frontend
├── frontend_app.py             # Manages frontend logic and layout
├── clf.pkl                     # Trained classification model
├── tfidf.pkl                   # TF-IDF vectorizer used in model
├── resumeats.py                # Resume parsing and analysis logic
├── UpdatedResumeDataSet.csv    # Labeled resume dataset
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration for deployment
├── video_assitance             # Voice/video support file
└── README.md                   # Project documentation 

## 🧠 How it Works

1. Resume Upload: Upload your resume (PDF/DOCX).
2. Data Extraction: Resume content is parsed using NLP tools.
3. Model Inference: TF-IDF and classifier predict role suitability.
4. Feedback/Assistance: Optional video or audio output for assistance.
