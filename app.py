from fastapi import FastAPI, Query
import pandas as pd
import re
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()
ua = UserAgent()
url_cache = {}
df = None
vectorizer = None
tfidf_matrix = None

async def fetch_review(session, url):
    if url in url_cache:
        return url_cache[url]
    
    headers = {"User-Agent": ua.random}
    try:
        async with session.get(url, headers=headers, timeout=10) as response:
            if response.status != 200:
                return "Failed to fetch review"
            text = await response.text()
            soup = BeautifulSoup(text, 'html.parser')
            review_element = soup.find('div', class_=lambda x: x and 'text' in x.lower())
            review_text = review_element.text.strip() if review_element else "No review available"
            url_cache[url] = review_text
            return review_text
    except Exception as e:
        return f"Failed to fetch review: {str(e)}"

async def process_urls(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_review(session, url) for url in urls]
        return await asyncio.gather(*tasks)

async def load_and_clean_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df["review"] = df["review"].fillna("")
        
        if "review" not in df.columns:
            raise ValueError("Error: 'review' column not found!")

        urls = [review for review in df["review"] if review.startswith("http")]
        
        if urls:
            fetched_reviews = await process_urls(urls)
            url_to_review = dict(zip(urls, fetched_reviews))
            df["review"] = df["review"].apply(lambda x: url_to_review.get(x, x))
        
        return df
    except Exception as e:
        return pd.DataFrame()

def preprocess_reviews(df):
    df['cleaned_review'] = df['review'].astype(str).str.lower().str.replace(r'\W+', ' ', regex=True)
    return df

def train_vectorizer(df):
    if df.empty or "cleaned_review" not in df:
        return None, None

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['cleaned_review'].dropna())
    
    return vectorizer, tfidf_matrix

def answer_query(df, vectorizer, tfidf_matrix, query):
    if df.empty or 'cleaned_review' not in df:
        return "No data available to process the query."

    query_vec = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    if similarity_scores.max() == 0:
        return "No relevant reviews found."
    
    best_match_idx = similarity_scores.argmax()
    return df.iloc[best_match_idx]['review']

@app.on_event("startup")
async def startup_event():
    global df, vectorizer, tfidf_matrix
    file_path = "movie_reviews.csv"
    
    df = await load_and_clean_data(file_path)
    df = preprocess_reviews(df)
    vectorizer, tfidf_matrix = train_vectorizer(df)

@app.get("/query/")
def get_review(query: str = Query(..., description="Ask about a movie")):
    query = re.sub(r'[^a-zA-Z0-9 ]', '', query.lower())
    return {"response": answer_query(df, vectorizer, tfidf_matrix, query)}
