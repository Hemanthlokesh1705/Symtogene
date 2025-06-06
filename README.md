# SymptoGene: Symptom-Based Genetic Disease Identifier 🧬

SymptoGene is an AI-powered web application designed to predict **genetic diseases** based on user-input symptoms. It combines machine learning with a clean, interactive interface to assist users in identifying potential genetic conditions and guiding them to appropriate healthcare resources.

## 🚀 Features

- 🔍 **Symptom-Based Disease Prediction** using trained ML models
- 💬 **AI Chatbot** to help users describe and refine their symptoms
- 👨‍⚕️ **Doctor Directory** showing specialists related to predicted conditions
- 📅 **Appointment Booking** with doctors directly from the platform
- 🔐 **User Login & Registration** for personalized experience and saved history
- 📊 **Graphical Results** for better visualization of disease predictions

## 🧠 Technologies Used

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python (Flask)
- **Machine Learning**: Scikit-learn, Pandas
- **Deployment**: (To be added: Render, Heroku, or local server)
- *(Optional)*: Google APIs for future chatbot voice/speech features

## 📁 Project Structure

SymptoGene/
├── static/ # CSS, JS, images
├── templates/ # HTML templates
├── model/
│ └── predictor.py # ML model logic
├── app.py # Flask main file
├── requirements.txt # Python dependencies
└── README.md # This file

bash
Copy
Edit

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/SymptoGene.git
   cd SymptoGene
Install dependencies:

pip install -r requirements.txt
Run the application:
python app.py
Open your browser:

http://127.0.0.1:5000/
📊 How It Works
User enters symptoms via form or chatbot.

Symptoms are processed and sent to the backend.

ML model predicts likely genetic diseases.

User sees predictions with disease info and can book appointments.
