from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager # ADD THIS LINE
from pydantic import BaseModel
import numpy as np
import pickle
import re
import os
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
import distance
from fuzzywuzzy import fuzz

# Download stopwords on startup
nltk.download('stopwords', quiet=True)

app = FastAPI(title="Quora Question Pairs Duplicate Detection API")

from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "="*55)
    print("🚀 APP IS LIVE! CLICK HERE: http://localhost:8000")
    print("="*55 + "\n")
    yield

app = FastAPI(title="Quora Question Pairs Duplicate Detection API")

# ADD THESE LINES TO ENABLE BROWSER ACCESS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows your HTML file to communicate with the API
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the saved Random Forest and CountVectorizer models
if os.path.exists('model.pkl') and os.path.exists('cv.pkl'):
    with open('model.pkl', 'rb') as f:
        rf = pickle.load(f)
    with open('cv.pkl', 'rb') as f:
        cv = pickle.load(f)
else:
    rf, cv = None, None

class QuestionPair(BaseModel):
    question1: str
    question2: str

def preprocess(q):
    q = str(q).lower().strip()
    
    # Replace certain special characters with their string equivalents
    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')
    q = q.replace('[math]', '')
    
    # Replacing some numbers with string equivalents
    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)
    
    # Decontracting words
    contractions = {
        "ain't": "am not", "aren't": "are not", "can't": "can not", "can't've": "can not have",
        "'cause": "because", "could've": "could have", "couldn't": "could not", "couldn't've": "could not have",
        "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
        "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", "he'd": "he would",
        "he'd've": "he would have", "he'll": "he will", "he'll've": "he will have", "he's": "he is",
        "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
        "i'd": "i would", "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have",
        "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have",
        "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us",
        "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not",
        "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
        "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
        "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
        "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
        "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
        "she'll've": "she will have", "she's": "she is", "should've": "should have",
        "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
        "so's": "so as", "that'd": "that would", "that'd've": "that would have",
        "that's": "that is", "there'd": "there would", "there'd've": "there would have",
        "there's": "there is", "they'd": "they would", "they'd've": "they would have",
        "they'll": "they will", "they'll've": "they will have", "they're": "they are",
        "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would",
        "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
        "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
        "what'll've": "what will have", "what're": "what are", "what's": "what is",
        "what've": "what have", "when's": "when is", "when've": "when have",
        "where'd": "where did", "where's": "where is", "where've": "where have",
        "who'll": "who will", "who'll've": "who will have", "who's": "who is",
        "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have",
        "won't": "will not", "won't've": "will not have", "would've": "would have",
        "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
        "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are",
        "y'all've": "you all have", "you'd": "you would", "you'd've": "you would have",
        "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"
    }
    
    q_decontracted = []
    for word in q.split():
        if word in contractions:
            word = contractions[word]
        q_decontracted.append(word)
        
    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")
    
    # Removing HTML tags
    q = BeautifulSoup(q, "html.parser").get_text()
    
    # Remove punctuations
    pattern = re.compile(r'\W')
    q = re.sub(pattern, ' ', q).strip()
    return q

def fetch_token_features(q1, q2):
    SAFE_DIV = 0.0001
    STOP_WORDS = stopwords.words("english")
    token_features = [0.0]*8
    
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features
        
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
    
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])
    
    common_word_count = len(q1_words.intersection(q2_words))
    common_stop_count = len(q1_stops.intersection(q2_stops))
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
    
    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    
    return token_features

def fetch_length_features(q1, q2):
    length_features = [0.0]*3
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features
        
    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
    length_features[1] = (len(q1_tokens) + len(q2_tokens)) / 2
    
    strs = list(distance.lcsubstrings(q1, q2))
    if len(strs) > 0:
        length_features[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1)
    return length_features

def fetch_fuzzy_features(q1, q2):
    fuzzy_features = [0.0]*4
    fuzzy_features[0] = fuzz.QRatio(q1, q2)
    fuzzy_features[1] = fuzz.partial_ratio(q1, q2)
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)
    fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)
    return fuzzy_features

def extract_features(q1, q2):
    q1 = preprocess(q1)
    q2 = preprocess(q2)
    
    # Basic Feature Engineering
    q1_len = len(q1)
    q2_len = len(q2)
    q1_num_words = len(q1.split(" "))
    q2_num_words = len(q2.split(" "))
    
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))
    word_common = len(w1 & w2)
    word_total = len(w1) + len(w2)
    word_share = round(word_common / word_total, 2) if word_total > 0 else 0.0
    
    # Advanced features
    token_features = fetch_token_features(q1, q2)
    length_features = fetch_length_features(q1, q2)
    fuzzy_features = fetch_fuzzy_features(q1, q2)
    
    # Combine engineered features (22 items total)
    engineered = [
        q1_len, q2_len, q1_num_words, q2_num_words, word_common, word_total, word_share
    ] + token_features + length_features + fuzzy_features
    
    # Bag of Words transformations
    q1_bow = cv.transform([q1]).toarray()[0]
    q2_bow = cv.transform([q2]).toarray()[0]
    
    # Create unified feature vector (22 + 3000 + 3000 = 6022 items)
    features_array = np.hstack((np.array(engineered), q1_bow, q2_bow)).reshape(1, -1)
    
    return features_array

@app.post("/predict")
def predict(pair: QuestionPair):
    if rf is None or cv is None:
        raise HTTPException(status_code=500, detail="Machine Learning model artifacts (model.pkl / cv.pkl) are missing!")
        
    # Extract complete feature set matching final_df structure
    features = extract_features(pair.question1, pair.question2)
    
    # Execute Model prediction
    prediction = rf.predict(features)[0]
    
    return {
        "question1": pair.question1,
        "question2": pair.question2,
        "is_duplicate": int(prediction)
    }


from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Mount the static folder for CSS
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the HTML file at the root URL
@app.get("/")
async def serve_frontend():
    return FileResponse('index.html')