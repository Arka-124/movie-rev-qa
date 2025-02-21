from fastapi import FastAPI, Query
import pandas as pd
import re
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from review_handler import fetch_review, process_urls, preprocess_reviews, train_vectorizer, answer_query, load_and_clean_data
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
ua = UserAgent()
url_cache = {}
df = None
vectorizer = None
tfidf_matrix = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
