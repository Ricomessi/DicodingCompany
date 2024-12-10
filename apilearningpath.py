from flask import Flask, request, jsonify
import joblib
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import requests
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Constants
KMEANS_MODEL_PATH = 'model/kmeans_model.pkl'
VECTORIZER_PATH = 'model/vectorizer.pkl'
DATA_PATH = "data/updatemergedata.csv"
GITHUB_API_URL = "https://api.github.com/users/{}/repos"
MIN_RECOMMENDATIONS = 6
SIMILARITY_THRESHOLD = 0.1
DEFAULT_USER_WEIGHT = 0.6
DEFAULT_GITHUB_WEIGHT = 0.4

# Load pre-trained models and vectorizer
kmeans = joblib.load(KMEANS_MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# Stopwords setup
ENGLISH_STOP_WORDS = set(stopwords.words('english'))
INDONESIAN_STOP_WORDS = set(stopwords.words('indonesian'))
ALL_STOP_WORDS = ENGLISH_STOP_WORDS.union(INDONESIAN_STOP_WORDS)

# Load and preprocess dataset
data = pd.read_csv(DATA_PATH)
data['Combined Summary'] = (
    data['Learning Path'] + ' ' +
    data['Learning Path Summary'] + ' ' +
    data['Course Name_x'] + ' ' +
    data['Course Summary']
)
data['Combined Summary'] = data['Combined Summary'].apply(
    lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in ALL_STOP_WORDS])
)

# Vectorizing the cleaned data for clustering
X = vectorizer.transform(data['Combined Summary'])

# Predict clusters
data['Cluster'] = kmeans.predict(X)


# Function to fetch GitHub keywords
def fetch_github_keywords(github_username):
    """Fetches keywords from GitHub repositories based on the given username."""
    url = GITHUB_API_URL.format(github_username)
    try:
        response = requests.get(url)
        response.raise_for_status()
        repos = response.json()

        if not repos:
            return ""

        keywords = []
        for repo in repos:
            text = f"{repo['name']} {repo['description'] or ''} {repo['language'] or ''}"
            cleaned_text = re.sub(r"[^a-zA-Z\s]", "", text).lower()
            tokens = word_tokenize(cleaned_text)
            filtered_tokens = [word for word in tokens if word not in ALL_STOP_WORDS]
            keywords.extend(filtered_tokens)

        unique_keywords = list(set(keywords))
        return " ".join(unique_keywords)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching GitHub data for {github_username}: {e}")
        return ""


# Recommendation function
def recommend_courses(user_skill, user_language, github_username, similarity_threshold=SIMILARITY_THRESHOLD, min_recommendations=MIN_RECOMMENDATIONS):
    """Generates course recommendations based on user skill, language, and GitHub profile."""
    # Fetch GitHub keywords
    github_keywords = fetch_github_keywords(github_username) if github_username else ""

    # Combine the user's skill, language, and GitHub keywords
    if any([user_skill, user_language, github_keywords]):
        combined_input = f"{user_skill or ''} {user_language or ''} {github_keywords or ''}".strip()
        user_vector = vectorizer.transform([combined_input])
        course_vectors = vectorizer.transform(data['Combined Summary'])

        # Calculate cosine similarity
        course_similarities = cosine_similarity(user_vector, course_vectors).flatten()
        data['Similarity'] = course_similarities
        filtered_courses = data[data['Similarity'] > similarity_threshold]
    else:
        # If no input is provided, return a fallback based on basic level courses
        data['Similarity'] = 0.0
        filtered_courses = data.copy()

    # If recommendations are less than the minimum required, return fallback courses
    if len(filtered_courses) < min_recommendations:
        fallback_courses = fallback_recommendations(min_recommendations)
        return {"recommendations": format_recommendations(fallback_courses)}

    # Sort and filter recommendations
    level_order = ['Dasar', 'Pemula', 'Menengah', 'Mahir', 'Profesional']
    level_dtype = CategoricalDtype(categories=level_order, ordered=True)
    filtered_courses['Level'] = filtered_courses['Level'].astype(level_dtype)
    filtered_courses = filtered_courses.sort_values(by=['Level', 'Similarity'], ascending=[True, False])
    filtered_courses = filtered_courses.drop_duplicates(subset=['Course Name_x'])

    # Select top recommendations
    final_recommendations = []
    for level in level_order:
        level_courses = filtered_courses[filtered_courses['Level'] == level]
        top_courses = level_courses.sort_values(by='Similarity', ascending=False).head(3)
        final_recommendations.extend(top_courses.to_dict(orient='records'))
    final_recommendations = final_recommendations[:10]

    return {"recommendations": format_recommendations(final_recommendations)}


# Helper function for fallback recommendations
def fallback_recommendations(min_recommendations):
    """Generates fallback recommendations if the primary recommendations are insufficient."""
    fallback_courses = data[(data['Level'] == 'Dasar') | (data['Level'] == 'Pemula')]
    fallback_courses = fallback_courses.sort_values(by=['Jumlah Enrollment'], ascending=False).head(min_recommendations)
    return fallback_courses


# Helper function to format recommendations into JSON
def format_recommendations(courses):
    """Formats the course data for JSON output."""
    return [
        {
            "title": course['Course Name_x'],
            "description": course['Course Summary'],
        }
        for _, course in pd.DataFrame(courses).iterrows()
    ]


@app.route('/recommend', methods=['POST'])
def get_recommendations():
    """API endpoint to get course recommendations."""
    data = request.get_json()

    user_skill = data.get('skill')
    user_language = data.get('language')
    github_username = data.get('github_username')
    similarity_threshold = data.get('similarity_threshold', SIMILARITY_THRESHOLD)

    # Validate inputs
    if not user_skill and not user_language and not github_username:
        logging.warning("No input provided, returning fallback courses.")
        fallback_courses = fallback_recommendations(MIN_RECOMMENDATIONS)
        return jsonify({"recommendations": format_recommendations(fallback_courses)})

    # Get and return recommendations
    recommendations = recommend_courses(user_skill, user_language, github_username, similarity_threshold)
    return jsonify(recommendations)


if __name__ == '__main__':
    app.run(debug=True)
