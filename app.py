import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import os
openai.api_key = os.environ.get("OPENAI_API_KEY")
import openai
openai.api_key = "YOUR_OPENAI_API_KEY"

df = pd.read_csv("movie_reviews.csv")
df = pd.read_csv("movie_reviews.csv")
print(df.columns)


import requests
from bs4 import BeautifulSoup

def remove_html_tags(text):
    if text and text.startswith("http"): 
        try:
            response = requests.get(text)
            response.raise_for_status() 
            soup = BeautifulSoup(response.content, "html.parser")
            review_element = soup.find("div", class_="review-text")
            if review_element:
                return review_element.get_text().strip()
            else:
                print(f"Review element not found at {text}")
                return "" 
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {text}: {e}")
            return ""
        except Exception as e:
            print(f"Error parsing {text}: {e}")
            return ""
    elif text: 
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()
    else: 
        return ""
df['review'] = df['review'].apply(remove_html_tags)


def remove_punctuation(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
    return text

df['review'] = df['review'].apply(remove_punctuation)


nltk.download('stopwords', quiet=True) # Download stopwords if you haven't already
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)

df['review'] = df['review'].apply(remove_stopwords)
all_sentences = []
movie_to_sentences = {}  # Dictionary to map movie titles to their sentences

for index, row in df.iterrows():
    movie_title = row['movie_title'] # Make sure your CSV has a movie_title column
    sentences = row['review'].split('.') # Split review into sentences
    for sentence in sentences:
        cleaned_sentence = sentence.strip()
        if cleaned_sentence: # Skip empty sentences
            all_sentences.append(cleaned_sentence)
            if movie_title not in movie_to_sentences:
                movie_to_sentences[movie_title] = []
            movie_to_sentences[movie_title].append(cleaned_sentence)
def get_relevant_sentences(movie_title, question):
    relevant_sentences = []
    if movie_title in movie_to_sentences:
        for sentence in movie_to_sentences[movie_title]:
            if any(keyword in sentence for keyword in question.split()): 
                relevant_sentences.append(sentence)
    return relevant_sentences
from sklearn.feature_extraction.text import TfidfVectorizer

def get_relevant_sentences(movie_title, question):
    if movie_title not in movie_to_sentences:
        return []

    sentences = movie_to_sentences[movie_title]
    vectorizer = TfidfVectorizer()
    sentence_vectors = vectorizer.fit_transform(sentences)
    question_vector = vectorizer.transform([question])
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(question_vector, sentence_vectors)
    top_indices = similarities.argsort()[0][::-1][:3] # Get top 3

    relevant_sentences = [sentences[i] for i in top_indices]

    return relevant_sentences
  # Replace with your actual key

def generate_answer(question, context):
    prompt = f"Question: {question}\nContext: {context}\nAnswer:"
    response = openai.Completion.create(
        engine="text-davinci-003",  # Or a suitable model
        prompt=prompt,
        max_tokens=150,  # Adjust as needed
    )
    return response.choices[0].text.strip()
   
    
