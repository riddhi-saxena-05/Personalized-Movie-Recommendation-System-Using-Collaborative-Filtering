# -*- coding: utf-8 -*-
"""
Movie Recommendation System Web App
"""

import os
import difflib
import pickle
import streamlit as st
import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "movies.csv")
MODEL_PATH = os.path.join(BASE_DIR, "trained_model.sav")

@st.cache_data(show_spinner=False)
def load_data_and_similarity():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"movies.csv not found at {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    selected_features = [
        'genres', 'keywords', 'overview', 'production_companies',
        'production_countries', 'spoken_languages', 'tagline', 'cast', 'director'
    ]

    for feature in selected_features:
        if feature in df.columns:
            df[feature] = df[feature].fillna('')
        else:
            df[feature] = ''
        
    vectorizer = TfidfVectorizer()
    df['descreption'] = (
        df['genres'] + ', ' +
        df['keywords'] + ', ' +
        df['overview'] + ', ' +
        df['production_companies'] + ', ' +
        df['production_countries'] + ', ' +
        df['spoken_languages'] + ', ' +
        df['tagline'] + ', ' +
        df['cast'] + ', ' +
        df['director']
    )
    feature_vectors = vectorizer.fit_transform(df['descreption'])
    similarity = cosine_similarity(feature_vectors)
    return df, similarity

def get_recommendations(input_movie, top_n=20):
    if not input_movie or not input_movie.strip():
        return [], None, "Please enter a movie title."

    df, similarity = load_data_and_similarity()
    list_of_all_titles = df['original_title'].dropna().tolist()
    find_close_match = difflib.get_close_matches(input_movie.strip(), list_of_all_titles, n=1, cutoff=0.2)

    if not find_close_match:
        return [], None, f"No close matches found for '{input_movie}'. Please check the spelling or try another movie."

    close_match = find_close_match[0]
    matched_rows = df[df.original_title.str.lower() == close_match.lower()]
    if matched_rows.empty:
        matched_rows = df[df.original_title == close_match]
    
    index_of_the_movie = matched_rows['index'].values[0]
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    recommended_movies = []
    i = 0
    for movie in sorted_similar_movies:
        index = movie[0]
        row = df[df['index'] == index]
        if not row.empty:
            movie_data = {
                'title': row['original_title'].values[0],
                'genres': row['genres'].values[0] if 'genres' in row else '',
                'vote_average': row['vote_average'].values[0] if 'vote_average' in row else 0,
                'director': row['director'].values[0] if 'director' in row else '',
                'cast': row['cast'].values[0] if 'cast' in row else '',
                'tagline': row['tagline'].values[0] if 'tagline' in row else '',
                'overview': row['overview'].values[0] if 'overview' in row else '',
                'similarity_score': movie[1]
            }
            recommended_movies.append(movie_data)
            i += 1
            if i >= top_n + 1:  # Include match + recommendations
                break

    return recommended_movies, close_match, None

def Movies_prediction(input_movie):
    """Backward-compatible function returning formatted string of recommendations."""
    recommendations, match, error = get_recommendations(input_movie, top_n=20)
    if error:
        return error

    liste = ' '
    i = 1
    for movie in recommendations[1:]:  # skip the exact input movie itself
        liste = '\n' + liste + str(i) + '. ' + str(movie['title']) + '\n\n'
        i += 1
        if i > 20:
            break
    return liste

def main():
    st.markdown("<h1 style='text-align: center; color: #E50914;'>🎬 MOVIES RECOMMENDATION SYSTEM</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray; font-size: 1.1em;'>Discover movies similar to your favorites using Machine Learning & Cosine Similarity</p>", unsafe_allow_html=True)
    st.divider()

    try:
        with st.spinner("Loading dataset and initializing recommendation engine..."):
            df, _ = load_data_and_similarity()
    except Exception as e:
        st.error(f"Error loading movie dataset: {e}")
        return

    all_titles = sorted(df['original_title'].dropna().unique().tolist())

    col1, col2 = st.columns([3, 1])
    with col1:
        search_mode = st.radio("Search Mode", ["Select from list / Autocomplete", "Type movie name"], horizontal=True)
        if search_mode == "Select from list / Autocomplete":
            default_index = all_titles.index("The Avengers") if "The Avengers" in all_titles else 0
            movie_input = st.selectbox("Choose a movie:", options=all_titles, index=default_index)
        else:
            movie_input = st.text_input("Enter your favorite movie name:", placeholder="e.g. Inception, Avatar, Spider-Man")

    with col2:
        top_k = st.slider("Number of recommendations", min_value=5, max_value=30, value=10, step=5)
        st.write("")
        recommend_btn = st.button("✨ Get Recommendations", use_container_width=True, type="primary")

    if recommend_btn or movie_input:
        if not movie_input:
            st.warning("Please enter or select a movie.")
            return

        with st.spinner("Finding best recommendations for you..."):
            recommendations, matched_title, error = get_recommendations(movie_input, top_n=top_k)

        if error:
            st.warning(error)
        elif recommendations:
            st.success(f"Showing recommendations based on closest match: **{matched_title}**")
            
            # Show top recommendations in organized cards/grid
            results = recommendations[1:top_k+1]  # Exclude the query movie itself
            
            for idx, movie in enumerate(results, 1):
                with st.expander(f"**{idx}. {movie['title']}** ⭐ {movie['vote_average']}/10", expanded=(idx <= 3)):
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        if movie['tagline']:
                            st.markdown(f"*{movie['tagline']}*")
                        if movie['overview']:
                            st.write(movie['overview'])
                    with c2:
                        if movie['genres']:
                            st.markdown(f"**🎭 Genres:** {movie['genres']}")
                        if movie['director']:
                            st.markdown(f"**🎬 Director:** {movie['director']}")
                        if movie['cast']:
                            st.markdown(f"**👥 Cast:** {movie['cast'][:100]}...")

if __name__ == '__main__':
    main()
    