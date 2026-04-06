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

## 💻 Local Development

If you have **Docker** installed, you can run this entire project with two commands:

1. **Build the Image:**
   ```bash
   docker build -t duplicate-predictor .

2. **Run The Container:**
   ```bash
   docker run -p 8000:8000 duplicate-predictor
