# 🔍 Is The Question Duplicate? 
### Semantic Similarity Detection using NLP & Machine Learning

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=for-the-badge&logo=docker)
![Render](https://img.shields.io/badge/Deploy-Render-430098?style=for-the-badge&logo=render)

A professional-grade, end-to-end Machine Learning application designed to identify duplicate question pairs. This tool solves the problem of information redundancy on platforms like Quora or StackOverflow by using advanced linguistic analysis.

---

## 🌐 Live Application
> **Try it here:** [isthequestionduplicate.onrender.com](https://duplicate-question-ai.onrender.com/)
> *(Note: Free-tier instances may take 30-60 seconds to spin up if they have been idle.)*

---

## 🚀 The NLP Pipeline
This project doesn't just look at word counts; it extracts **22 sophisticated features** to understand the "meaning" behind the text:

* **Fuzzy String Matching:** Uses Levenshtein distance (FuzzyWuzzy) to catch typos and partial matches.
* **Token Analysis:** Calculates word commonality ratios and stop-word filtering.
* **Length Statistics:** Measures absolute differences and longest common substrings.
* **Vectorization:** Uses a pre-trained `CountVectorizer` to transform raw text into numerical data for the model.

---

## 🛠️ Technical Architecture

### **Backend & ML**
* **Model:** Random Forest Classifier (Optimized for high precision).
* **Framework:** FastAPI (Asynchronous request handling).
* **Preprocessing:** NLTK, BeautifulSoup (HTML cleaning), and Regex.

### **Frontend**
* **Design:** Modern Glassmorphism UI.
* **UX:** Interactive JavaScript state management with real-time feedback animations.

### **DevOps**
* **Containerization:** Multi-stage Docker build for a lightweight production image.
* **Deployment:** Automated CI/CD via GitHub and Render.

---

## ⚙️ How It All Works (The Deep Dive)

This project follows a modern, decoupled architecture that bridges the gap between raw data and a user-friendly interface.

### 1. Data Processing & ML (`main.py`, `model.pkl`, `cv.pkl`)
* **The Brain:** The `model.pkl` contains a **Random Forest Classifier** trained on the Quora Question Pairs dataset.
* **The Interpreter:** The `cv.pkl` (CountVectorizer) converts user-inputted text into numerical vectors that the model can understand.
* **The Engine:** `main.py` isn't just a server; it handles the heavy lifting of **Feature Engineering**—calculating Fuzzy ratios, token similarities, and length differences in real-time.

### 2. The API Layer (`FastAPI`)
* **Asynchronous Processing:** Uses Python's `async` capabilities to handle multiple prediction requests simultaneously without slowing down.
* **Automatic Documentation:** FastAPI automatically generates interactive API docs (available at `/docs` when running locally).

### 3. Frontend & UX (`index.html`, `static/`)
* **Glassmorphism Design:** A modern, semi-transparent UI that uses CSS back-drop filters for a premium feel.
* **State Management:** Vanilla JavaScript handles the "Predict" button states (Loading... -> Result) without requiring a page refresh (AJAX/Fetch API).

### 4. Portability (`Dockerfile`, `requirements.txt`)
* **The Blueprint:** The `Dockerfile` ensures that the app runs in the exact same Python 3.9 environment regardless of whether it's on your laptop or a cloud server.
* **Dependency Management:** `requirements.txt` locks in the versions of libraries like `scikit-learn` and `fuzzywuzzy` to prevent "it works on my machine" bugs.

---

## 💻 Local Development

If you have **Docker** installed, you can run this entire project with two commands:

1. **Build the Image:**
   ```bash
   docker build -t duplicate-predictor .

2. **Run The Container:**
   ```bash
   docker run -p 8000:8000 duplicate-predictor
