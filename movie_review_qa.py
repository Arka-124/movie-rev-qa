import pandas as pd
import re
import requests
import time
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    df["review"] = df["review"].fillna("") 
    ua = UserAgent()
    
    def extract_first_review(url):
        headers = {"User-Agent": ua.random}  # Mimic a real user
        try:
            time.sleep(3)  # Add delay to avoid detection
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract the first review from the IMDb page
            review_element = soup.find('div', class_=lambda x: x and 'text' in x.lower())
            return review_element.text.strip() if review_element else "No review available"
        except requests.RequestException:
            return "Failed to fetch review"
    
    def clean_review(text):
        if isinstance(text, str):
            if text.startswith("http"):
                return extract_first_review(text)
            return text  # Keep actual review text unchanged
        return ""
    
    df["review"] = df["review"].apply(clean_review)
    return df

def preprocess_reviews(df):
    df['cleaned_review'] = df['review'].astype(str).str.lower().replace(r'[^a-zA-Z0-9 ]', '', regex=True)
    return df

def train_vectorizer(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['cleaned_review'].dropna())
    return vectorizer, tfidf_matrix

def answer_query(df, vectorizer, tfidf_matrix, query):
    if df.empty or 'cleaned_review' not in df:
        return "No data available to process the query."
    
    query_vec = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    best_match_idx = similarity_scores.argmax()
    return df.iloc[best_match_idx]['review'] if similarity_scores[best_match_idx] > 0 else "No relevant reviews found."

if __name__ == "__main__":
    file_path = "movie_reviews.csv"
    try:
        df = load_and_clean_data(file_path)
        df = preprocess_reviews(df)
        vectorizer, tfidf_matrix = train_vectorizer(df)
    except Exception as e:
        print(f"Error loading or processing data: {e}")
        df = pd.DataFrame()
    
    while True:
        user_query = input("Ask about a movie (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        response = answer_query(df, vectorizer, tfidf_matrix, user_query)
        print("\nAnswer:", response, "\n")
