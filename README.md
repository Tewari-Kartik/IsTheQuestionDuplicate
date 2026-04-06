# ?? Is The Question Duplicate? 
### End-to-End NLP Machine Learning Web Application

A sophisticated tool built to detect semantic overlap between question pairs. Using a **Random Forest Classifier** and advanced **Natural Language Processing (NLP)**, this app determines if two questions share the same "intent"—even if they use different wording.

---

## ?? Live Demo
[**Click Here to View the App on Render**](https://duplicate-question-ai.onrender.com/)  

---

## ? Key Features
* **Feature Engineering:** Extracts 22 high-impact features including Token Features, Length Features, and Fuzzy String Matching.
* **FastAPI Backend:** A high-performance asynchronous API for real-time model inference.
* **Dockerized Deployment:** Fully containerized using **Docker** for portability.
* **Modern UI:** A custom-built **Glassmorphism** interface with CSS animations.

---

## ??? Tech Stack
| Category | Tools & Libraries |
| :--- | :--- |
| **Machine Learning** | Python, Scikit-Learn, NumPy, Pandas |
| **NLP** | NLTK, BeautifulSoup4, FuzzyWuzzy, Distance |
| **Backend** | FastAPI, Uvicorn, Pydantic |
| **Frontend** | HTML5, CSS3, JavaScript |
| **DevOps** | Docker, Docker Hub, Render |

---

## ?? How to Run Locally

1. **Clone the Repo:**
   \\\ash
   git clone https://github.com/Tewari-Kartik/IsTheQuestionDuplicate.git
   cd IsTheQuestionDuplicate
   \\\

2. **Build & Run with Docker:**
   \\\ash
   docker build -t duplicate-predictor .
   docker run -p 8000:8000 duplicate-predictor
   \\\

---

## ?? Project Structure
* \main.py\: FastAPI application logic and NLP feature extraction.
* \model.pkl\ & \cv.pkl\: Pre-trained model weights and vectorizer.
* \Dockerfile\: Containerization blueprint.

---

## ?? Contact
**Kartik Tewari** [GitHub Profile](https://github.com/Tewari-Kartik)
