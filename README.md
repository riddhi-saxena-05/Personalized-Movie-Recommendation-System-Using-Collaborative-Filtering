# Personalized Movie Recommendation System

A machine learning-powered movie recommendation system deployed with Streamlit. The system recommends movies based on content similarity and metadata (genres, keywords, cast, director, and overview) using TF-IDF vectorization and Cosine Similarity.

## 📌 Table of Contents
- [Features](#-features)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Model Overview](#-model-overview)
- [Dataset](#-dataset)

---

## ✨ Features
- 🔍 **Interactive Movie Search & Autocomplete**: Select or search for any movie in the database.
- 🎯 **Accurate Recommendations**: Uses TF-IDF feature extraction & Cosine Similarity across genres, directors, cast, taglines, and descriptions.
- 📊 **Rich Movie Details**: Displays overview, director, cast, genres, and ratings for recommended movies.
- ⚡ **Streamlit UI**: Clean, responsive interface for instant exploration.

---

## 🚀 Getting Started

To run this recommendation system on your local machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/riddhi-saxena-05/Personalized-Movie-Recommendation-System-Using-Collaborative-Filtering.git
   cd Personalized-Movie-Recommendation-System-Using-Collaborative-Filtering
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application:**
   ```bash
   streamlit run "web app.py"
   ```

4. Open your browser at `http://localhost:8501`.

---

## 📖 Usage
1. Choose between **Autocomplete selection** or **Text Search**.
2. Pick or enter your favorite movie title.
3. Adjust the number of recommendations you want.
4. Click **Get Recommendations** to discover similar movies with full details!

---

## 🧠 Model Overview
The recommendation engine processes multiple textual and metadata features from each movie (`genres`, `keywords`, `overview`, `cast`, `director`, `tagline`, `production_companies`, and `spoken_languages`). It computes feature vectors using `TfidfVectorizer` and measures cosine distance across all movies to find the highest similarity matches.

---

## 📁 Dataset
The project utilizes the TMDB / MovieLens dataset containing movie metadata, ratings, cast, and overview information.
