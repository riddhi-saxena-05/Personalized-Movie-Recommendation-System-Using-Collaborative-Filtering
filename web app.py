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

def extract_unique_genres(df):
    """Extract distinct genres dynamically from the dataset."""
    if 'genres' not in df.columns:
        return []
    
    known_multiword = ['Science Fiction', 'TV Movie']
    genres_set = set()
    for entry in df['genres'].dropna():
        text = str(entry).strip()
        for multi in known_multiword:
            if multi.lower() in text.lower():
                genres_set.add(multi)
                text = text.replace(multi, '')
        for word in text.split():
            word = word.strip()
            if word:
                genres_set.add(word)
    return sorted(list(genres_set))

def matches_genre(movie_genres_str, target_genre):
    """Check if movie contains target genre, supporting multi-genre strings."""
    if not target_genre or target_genre == "All Genres":
        return True
    if not movie_genres_str or pd.isna(movie_genres_str):
        return False
    return target_genre.lower() in str(movie_genres_str).lower()

def get_recommendations(input_movie, top_n=20, selected_genre=None):
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
    for movie in sorted_similar_movies:
        index = movie[0]
        row = df[df['index'] == index]
        if not row.empty:
            movie_genres = row['genres'].values[0] if 'genres' in row else ''
            
            # Apply genre filter if a specific genre is selected (keep index 0 match or filter candidates)
            if selected_genre and selected_genre != "All Genres" and len(recommended_movies) > 0:
                if not matches_genre(movie_genres, selected_genre):
                    continue

            movie_data = {
                'title': row['original_title'].values[0],
                'genres': movie_genres,
                'vote_average': row['vote_average'].values[0] if 'vote_average' in row else 0,
                'director': row['director'].values[0] if 'director' in row else '',
                'cast': row['cast'].values[0] if 'cast' in row else '',
                'tagline': row['tagline'].values[0] if 'tagline' in row else '',
                'overview': row['overview'].values[0] if 'overview' in row else '',
                'similarity_score': movie[1]
            }
            recommended_movies.append(movie_data)
            if len(recommended_movies) >= top_n + 1:  # Include match + recommendations
                break

    return recommended_movies, close_match, None

def get_movies_by_genre(selected_genre, top_n=20):
    """Retrieve top movies belonging to a selected genre when no seed movie is entered."""
    if not selected_genre or selected_genre == "All Genres":
        return [], "Please select a specific genre to browse."
    
    df, _ = load_data_and_similarity()
    filtered_df = df[df['genres'].apply(lambda g: matches_genre(g, selected_genre))]
    
    if filtered_df.empty:
        return [], f"No movies found for genre '{selected_genre}'."
    
    if 'popularity' in filtered_df.columns:
        sorted_df = filtered_df.sort_values(by='popularity', ascending=False)
    elif 'vote_average' in filtered_df.columns:
        sorted_df = filtered_df.sort_values(by='vote_average', ascending=False)
    else:
        sorted_df = filtered_df
        
    results = []
    for _, row in sorted_df.head(top_n).iterrows():
        movie_data = {
            'title': row['original_title'],
            'genres': row['genres'] if 'genres' in row else '',
            'vote_average': row['vote_average'] if 'vote_average' in row else 0,
            'director': row['director'] if 'director' in row else '',
            'cast': row['cast'] if 'cast' in row else '',
            'tagline': row['tagline'] if 'tagline' in row else '',
            'overview': row['overview'] if 'overview' in row else '',
            'similarity_score': 1.0
        }
        results.append(movie_data)
    return results, None

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
    available_genres = ["All Genres"] + extract_unique_genres(df)

    col1, col2 = st.columns([3, 1])
    with col1:
        search_mode = st.radio(
            "Search & Recommendation Mode",
            ["Select from list / Autocomplete", "Type movie name", "Browse by Genre Only"],
            horizontal=True
        )

        movie_input = ""
        if search_mode == "Select from list / Autocomplete":
            default_index = all_titles.index("The Avengers") if "The Avengers" in all_titles else 0
            movie_input = st.selectbox("Choose a movie:", options=all_titles, index=default_index)
        elif search_mode == "Type movie name":
            movie_input = st.text_input("Enter your favorite movie name:", placeholder="e.g. Inception, Avatar, Spider-Man")

        # Genre Filter UI
        selected_genre = st.selectbox("🎭 Filter by Genre:", options=available_genres, index=0)

    with col2:
        top_k = st.slider("Number of recommendations", min_value=5, max_value=30, value=10, step=5)
        st.write("")
        recommend_btn = st.button("✨ Get Recommendations", use_container_width=True, type="primary")

    # Trigger action
    if recommend_btn or (movie_input and search_mode != "Browse by Genre Only") or (search_mode == "Browse by Genre Only" and selected_genre != "All Genres"):
        # Case 1: Browse by genre only
        if search_mode == "Browse by Genre Only":
            if selected_genre == "All Genres":
                st.info("Please select a specific genre from the dropdown to browse movies.")
                return

            with st.spinner(f"Fetching top movies in '{selected_genre}' genre..."):
                genre_movies, error = get_movies_by_genre(selected_genre, top_n=top_k)

            if error:
                st.warning(error)
            elif genre_movies:
                st.success(f"Showing top **{len(genre_movies)}** movies in **{selected_genre}** genre:")
                for idx, movie in enumerate(genre_movies, 1):
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

        # Case 2: Movie recommendation (with or without genre filter)
        else:
            if not movie_input:
                st.warning("Please enter or select a movie.")
                return

            genre_text = f" (filtered by genre: **{selected_genre}**)" if selected_genre != "All Genres" else ""
            with st.spinner(f"Finding best recommendations{genre_text}..."):
                recommendations, matched_title, error = get_recommendations(
                    movie_input,
                    top_n=top_k,
                    selected_genre=selected_genre
                )

            if error:
                st.warning(error)
            elif recommendations:
                results = recommendations[1:top_k+1]  # Exclude seed movie itself
                
                if not results:
                    st.warning(f"No similar movies found for **{matched_title}** under the genre **{selected_genre}**. Try selecting **All Genres** or a different genre.")
                else:
                    if selected_genre != "All Genres":
                        st.success(f"Showing top {len(results)} recommendations for **{matched_title}** matching genre **{selected_genre}**:")
                    else:
                        st.success(f"Showing top {len(results)} recommendations based on closest match: **{matched_title}**")
                    
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
    