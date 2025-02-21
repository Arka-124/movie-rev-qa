import aiohttp
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ua = UserAgent()
url_cache = {}

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
    except Exception:
        return "Failed to fetch review"

async def process_urls(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_review(session, url) for url in urls]
        return await asyncio.gather(*tasks)

def preprocess_reviews(df):
    df['cleaned_review'] = df['review'].astype(str).str.lower().str.replace(r'\W+', ' ', regex=True)
    return df

def train_vectorizer(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['cleaned_review'].dropna())
    return vectorizer, tfidf_matrix

def answer_query(df, vectorizer, tfidf_matrix, query):
    if df.empty or 'cleaned_review' not in df:
        return "No relevant reviews found."
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
