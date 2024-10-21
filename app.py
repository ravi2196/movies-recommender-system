from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

app = Flask(__name__)

# Load the dataset
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
movies = movies.merge(credits, on='title')

# Data preprocessing (same steps as before)
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)

def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(lambda x: convert(x)[0:3])

def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L

movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split())

movies['genres'] = movies['genres'].apply(lambda x: x if isinstance(x, list) else [])
movies['keywords'] = movies['keywords'].apply(lambda x: x if isinstance(x, list) else [])
movies['cast'] = movies['cast'].apply(lambda x: x if isinstance(x, list) else [])
movies['crew'] = movies['crew'].apply(lambda x: x if isinstance(x, list) else [])

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))

new = movies[['movie_id', 'title', 'tags']]

cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new['tags']).toarray()

similarity = cosine_similarity(vector)

def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movies = []
    for i in distances[1:6]:
        recommended_movies.append(new.iloc[i[0]].title)
    return recommended_movies

@app.route('/')
def home():
    # Pass the movie titles to the template
    movie_titles = new['title'].values
    return render_template('index.html', movie_titles=movie_titles)

@app.route('/recommend', methods=['POST'])
def recommend_movies():
    movie = request.form['movie_name']
    recommendations = recommend(movie)
    # Pass the movie titles to the template
    movie_titles = new['title'].values
    return render_template('index.html', movie=movie, recommendations=recommendations, movie_titles=movie_titles)

if __name__ == '__main__':
    app.run(debug=True)
