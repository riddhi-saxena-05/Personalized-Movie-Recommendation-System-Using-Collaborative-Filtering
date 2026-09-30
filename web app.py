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
    page_title="CineMatch",
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
    # Inject Custom CSS for Modern, Polished UI/UX
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background-color: #F7F7FC !important;
        color: #202044 !important;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1120px !important;
    }

    /* Hero Header */
    .hero-container {
        text-align: center;
        margin-bottom: 2rem;
    }
    .hero-title {
        font-size: 2.35rem;
        font-weight: 800;
        color: #202044;
        letter-spacing: -0.025em;
        margin-bottom: 0.35rem;
        line-height: 1.2;
    }
    .hero-title .brand-highlight {
        color: #5B4BDB;
        background: linear-gradient(135deg, #5B4BDB 0%, #8B7CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #585779;
        max-width: 620px;
        margin: 0 auto;
        line-height: 1.5;
        font-weight: 400;
    }

    /* Section & Label Styling */
    .stSelectbox label, .stTextInput label, .stSlider label, .stRadio label {
        color: #202044 !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        letter-spacing: -0.01em;
    }

    /* Input Fields & Select Boxes */
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #E6E5F5 !important;
        border-radius: 12px !important;
        color: #202044 !important;
        box-shadow: 0 2px 8px rgba(32, 32, 68, 0.03) !important;
        transition: all 0.2s ease !important;
    }

    div[data-baseweb="input"] > div:hover,
    div[data-baseweb="select"] > div:hover {
        border-color: #8B7CF6 !important;
    }

    div[data-baseweb="input"] > div:focus-within,
    div[data-baseweb="select"] > div:focus-within {
        border-color: #5B4BDB !important;
        box-shadow: 0 0 0 3px rgba(91, 75, 219, 0.14) !important;
    }

    /* Radio Group Card */
    div[data-testid="stRadio"] {
        background-color: #FFFFFF;
        padding: 10px 14px;
        border-radius: 12px;
        border: 1px solid #E6E5F5;
        box-shadow: 0 2px 8px rgba(32, 32, 68, 0.03);
        margin-bottom: 12px;
    }
    div[data-testid="stRadio"] > div {
        gap: 1.2rem;
    }
    div[data-testid="stRadio"] label span {
        color: #202044 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }

    /* Slider Customization */
    div[data-testid="stSlider"] {
        background-color: #FFFFFF;
        padding: 12px 16px;
        border-radius: 12px;
        border: 1px solid #E6E5F5;
        box-shadow: 0 2px 8px rgba(32, 32, 68, 0.03);
    }
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: #5B4BDB !important;
        border-color: #5B4BDB !important;
        box-shadow: 0 2px 6px rgba(91, 75, 219, 0.3) !important;
    }

    /* Buttons */
    .stButton > button,
    button[kind="primary"] {
        background: linear-gradient(135deg, #5B4BDB 0%, #4D3DCB 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 0.98rem !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 0.65rem 1.3rem !important;
        box-shadow: 0 4px 16px rgba(91, 75, 219, 0.32) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
    }

    .stButton > button:hover,
    button[kind="primary"]:hover {
        background: linear-gradient(135deg, #4A3AC5 0%, #3F2EB8 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 22px rgba(91, 75, 219, 0.42) !important;
    }

    .stButton > button:active,
    button[kind="primary"]:active {
        transform: translateY(0) !important;
    }

    /* Expander / Recommendation Result Cards */
    div[data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E6E5F5 !important;
        border-radius: 14px !important;
        box-shadow: 0 3px 12px rgba(32, 32, 68, 0.04) !important;
        margin-bottom: 12px !important;
        transition: all 0.2s ease !important;
    }

    div[data-testid="stExpander"]:hover {
        border-color: #8B7CF6 !important;
        box-shadow: 0 6px 20px rgba(91, 75, 219, 0.08) !important;
    }

    div[data-testid="stExpander"] summary {
        font-weight: 700 !important;
        color: #202044 !important;
        font-size: 1.02rem !important;
        padding: 0.8rem 1rem !important;
    }

    /* Alerts and Feedback Banners */
    div[data-testid="stAlert"] {
        border-radius: 12px !important;
        border: 1px solid #E6E5F5 !important;
        box-shadow: 0 2px 8px rgba(32, 32, 68, 0.03) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Hero Banner
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">🎬 Movie <span class="brand-highlight">Recommendation</span> System</h1>
        <p class="hero-subtitle">Discover personalized movie recommendations powered by Machine Learning and Cosine Similarity</p>
    </div>
    """, unsafe_allow_html=True)

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
    