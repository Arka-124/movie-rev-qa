# Movie Review QA API

## Issues
Relevant dataset not found 

## Overview
This project is a FastAPI-based application that allows users to query movie reviews. It processes movie review data, fetches missing reviews from external URLs, and uses TF-IDF and cosine similarity to provide relevant responses to user queries.

## Features
- Fetches movie reviews from a CSV file
- Scrapes reviews from URLs if necessary
- Cleans and preprocesses text data
- Uses TF-IDF and cosine similarity to match user queries with relevant reviews
- Provides an API endpoint for querying reviews

## Installation
1. Clone the repository:
   ```sh
   git clone <repository_url>
   cd movie-rev-qa
   ```

2. Create and activate a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate  # On Windows
   ```

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
1. Ensure the `movie_reviews.csv` file is in the project directory.
2. Run the FastAPI application:
   ```sh
   uvicorn app:app --reload
   ```
3. Access the API at: [http://127.0.0.1:8000](http://127.0.0.1:8000)
4. Query movie reviews using:
   ```sh
   http://127.0.0.1:8000/query/?query=<your_movie_question>
   ```

## API Endpoints
- `GET /query/` - Query the movie review database.
  - **Parameters:** `query` (str) - The question about a movie review.
  - **Response:** JSON object with the most relevant review.

## Dependencies
- FastAPI
- Pandas
- Aiohttp
- BeautifulSoup
- Fake-UserAgent
- Scikit-learn

## License
This project is licensed under the MIT License.
