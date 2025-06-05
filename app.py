from flask import Flask, request, jsonify, render_template
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import pickle

app = Flask(__name__)

# Load dataset and model
movies = pd.read_csv('movies_metadata_processed.csv')  # from Colab

# If no poster_url column, create placeholder poster URLs for demo
if 'poster_url' not in movies.columns:
    movies['poster_url'] = movies['title'].apply(
        lambda t: f"https://via.placeholder.com/150x225?text={t.replace(' ', '+')}"
    )

with open('knn_model.pkl', 'rb') as f:
    knn = pickle.load(f)

@app.route('/')
def home():
    genres = [col.replace('genre_', '') for col in movies.columns if col.startswith('genre_')]
    countries = [col.replace('country_', '') for col in movies.columns if col.startswith('country_')]
    return render_template('index.html', genres=genres, countries=countries)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.form
    input_vector = {}

    # Initialize all features with 0
    for col in movies.columns:
        if col != 'title' and col != 'poster_url':
            input_vector[col] = 0

    # Set user-selected features
    input_vector['genre_' + data['genre']] = 1
    input_vector['country_' + data['country']] = 1
    input_vector['age'] = int(data['age'])

    user_df = pd.DataFrame([input_vector])
    user_df = user_df[movies.columns.drop(['title', 'poster_url'])]

    distances, indices = knn.kneighbors(user_df)
    rec_movies = movies.iloc[indices[0]]

    # Prepare response with title, genres, and poster_url
    recommendations = []
    for _, row in rec_movies.iterrows():
        # Extract genres from columns with prefix 'genre_'
        movie_genres = [col.replace('genre_', '') for col in movies.columns if col.startswith('genre_') and row[col] == 1]
        recommendations.append({
            'title': row['title'],
            'genres': movie_genres,
            'poster_url': row['poster_url'] if 'poster_url' in row else f"https://via.placeholder.com/150x225?text={row['title'].replace(' ', '+')}"
        })

    return jsonify(recommendations)

@app.route('/autocomplete')
def autocomplete():
    q = request.args.get('q', '').lower()
    suggestions = []
    if q:
        suggestions = movies[movies['title'].str.lower().str.contains(q)]['title'].tolist()[:10]
    return jsonify(suggestions)

if __name__ == '__main__':
    app.run(debug=True)
