import os
import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, render_template, request
import PyPDF2
import spacy
from collections import defaultdict

# Loading the environment variables
load_dotenv()

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY", "AIzaSyCE7cn7GjkxGSCYLFOBz0Pw-_E-Q331YtM")
genai.configure(api_key=api_key)

# Initialize Flask app

app = Flask(__name__)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to summarize text using spaCy
def summarize_text(text, max_sentences=3):
    # Process the text with spaCy
    doc = nlp(text)
    
    # Create a frequency dictionary for each word
    word_frequencies = defaultdict(int)
    for token in doc:
        if not token.is_stop and not token.is_punct:
            word_frequencies[token.text.lower()] += 1
    
    # Score sentences based on word frequencies
    sentence_scores = defaultdict(int)
    for sent in doc.sents:
        for word in sent:
            if word.text.lower() in word_frequencies:
                sentence_scores[sent] += word_frequencies[word.text.lower()]
    
    # Sort sentences by score and select the top N sentences
    sorted_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    summary = " ".join([str(sentence) for sentence in sorted_sentences[:max_sentences]])
    
    return summary

# Function to generate policy using Gemini
def generate_policy(scenario):
    prompt = f"Generate policy for the following scenario: {scenario}"
    response = genai.GenerativeModel("gemini-1.5-pro-latest").generate_content(prompt)
    return response.text

# Home route
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Handle file upload for summarization
        if "file" in request.files:
            file = request.files["file"]
            if file.filename != "":
                if file.filename.endswith(".pdf"):
                    reader = PyPDF2.PdfReader(file)
                    text = "".join([page.extract_text() for page in reader.pages])
                else:
                    text = file.read().decode("utf-8")
                summary = summarize_text(text)
                return render_template("index.html", summary=summary)
        
        # Handle policy generation
        if "scenario" in request.form:
            scenario = request.form["scenario"]
            policy = generate_policy(scenario)
            return render_template("index.html", policy=policy)
    
    return render_template("index.html")

# Run the app
if __name__ == "__main__":
    app.run(debug=True)