import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import ast
from datetime import datetime

# Set page config
st.set_page_config(page_title="Movie Recommendation & Rating System", layout="wide")

# Custom CSS for styling
def apply_custom_theme():
    st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    h1, h2, h3 {
        color: #ff4b4b !important;
    }
    
    .css-1d391kg, .css-12oz5g7 {
        background-color: #1a1a1a;
    }
    
    .stTextInput > div > div > input, .stNumberInput > div > div > input {
        background-color: #2b2b2b;
        color: white;
        border-color: #ff4b4b;
    }
    
    .stSelectbox > div > div > div {
        background-color: #2b2b2b;
        color: white;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #ff4b4b, #ff8080);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #ff8080, #ff4b4b);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.2);
        transform: translateY(-2px);
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #2b2b2b, #1a1a1a);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #ff4b4b;
        margin-bottom: 10px;
    }
    
    .movie-title {
        color: #ff4b4b;
        font-size: 18px;
        font-weight: bold;
    }
    
    .movie-info {
        color: #cccccc;
        font-size: 14px;
    }
    
    .rating-high {
        color: #00ff00;
        font-weight: bold;
    }
    
    .rating-medium {
        color: #ffff00;
        font-weight: bold;
    }
    
    .rating-low {
        color: #ff0000;
        font-weight: bold;
    }
    
    .predicted-rating {
        background: linear-gradient(135deg, #ff4b4b, #ff8080);
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: bold;
        display: inline-block;
        margin: 5px 0;
    }
    
    .prediction-tab {
        background: linear-gradient(135deg, #2b2b2b, #1a1a1a);
        padding: 25px;
        border-radius: 15px;
        border: 2px solid #ff4b4b;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Apply the custom theme
apply_custom_theme()

# Load XGBoost model components
@st.cache_resource
def load_xgboost_model():
# Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the project root directory (parent of script_dir)
    project_root = os.path.dirname(script_dir)
    
    try:
        # Construct path relative to script location
        model_path = os.path.join(script_dir, 'movie_rating_xgboost.pkl')
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        return (
            model_data['model'],
            model_data['scaler'],
            model_data['vectorizer'],
            model_data['feature_columns'],
            model_data['is_trained']
        )
    except Exception as e:
        st.error(f"Error loading XGBoost model from {model_path}: {str(e)}")
        return None, None, None, None, False

# Load recommendation model components
@st.cache_resource
def load_recommendation_model():
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the project root directory (parent of script_dir)
    project_root = os.path.dirname(script_dir)
    
    try:
        # Load the TF-IDF vectorizer
        with open(os.path.join(project_root, 'model', 'tfidf_vectorizer.pkl'), 'rb') as f:
            tfidf = pickle.load(f)
        
        # Load the cosine similarity matrix
        with open(os.path.join(project_root, 'model', 'cosine_sim.pkl'), 'rb') as f:
            cosine_sim = pickle.load(f)
        
        # Load the indices mapping
        with open(os.path.join(project_root, 'model', 'indices.pkl'), 'rb') as f:
            indices = pickle.load(f)
        
        # Load the movie data
        data = pd.read_pickle(os.path.join(project_root, 'model', 'movie_data.pkl'))
        
        return tfidf, cosine_sim, indices, data
    except Exception as e:
        st.error(f"Error loading recommendation model components: {str(e)}")
        # Try to show what files are available
        if os.path.exists('model'):
            st.info(f"Files in model directory: {os.listdir('model')}")
        else:
            st.error("Model directory not found!")
        return None, None, None, None

# Helper functions for XGBoost predictions
def parse_json_features(text):
    """Parse JSON-like strings from TMDB data"""
    try:
        if pd.isna(text) or text == '':
            return []
        parsed = ast.literal_eval(str(text))
        if isinstance(parsed, list):
            return [item['name'] for item in parsed if 'name' in item]
        return []
    except:
        return []

def get_director(crew_str):
    """Extract director from crew data"""
    try:
        if pd.isna(crew_str) or crew_str == '':
            return 'Unknown'
        crew = ast.literal_eval(str(crew_str))
        for person in crew:
            if person.get('job') == 'Director':
                return person.get('name', 'Unknown')
    except:
        pass
    return 'Unknown'

def prepare_movie_features(movie_data, vectorizer, scaler, feature_columns):
    """Prepare features for a single movie prediction"""
    try:
        # Parse JSON columns
        genres_list = parse_json_features(movie_data.get('genres', '[]'))
        cast_list = parse_json_features(movie_data.get('cast', '[]'))
        director = get_director(movie_data.get('crew', '[]'))
        
        # Create text features
        genres_str = ' '.join(genres_list[:3])
        cast_str = ' '.join(cast_list[:5])
        combined_text = f"{genres_str} {cast_str} {director}".strip()
        
        # Create numeric features
        budget = float(movie_data.get('budget', 0))
        revenue = float(movie_data.get('revenue', 0))
        runtime = float(movie_data.get('runtime', 90))
        popularity = float(movie_data.get('popularity', 1))
        vote_count = float(movie_data.get('vote_count', 10))
        
        profit = revenue - budget
        roi = profit / budget if budget > 0 else 0
        log_budget = np.log1p(budget)
        log_revenue = np.log1p(revenue)
        log_popularity = np.log1p(popularity)
        log_vote_count = np.log1p(vote_count)
        num_genres = len(genres_list)
        num_cast = len(cast_list)
        
        # Create feature array
        numeric_features = np.array([
            log_budget, log_revenue, runtime, log_popularity,
            log_vote_count, profit, roi, num_genres, num_cast
        ]).reshape(1, -1)
        
        # Transform text features
        text_features = vectorizer.transform([combined_text]).toarray()
        
        # Combine features
        X_combined = np.hstack([numeric_features, text_features])
        
        # Scale numeric features only
        X_scaled = X_combined.copy()
        X_scaled[:, :len(feature_columns)] = scaler.transform(numeric_features)
        
        return X_scaled
        
    except Exception as e:
        st.error(f"Error preparing features: {str(e)}")
        return None

def predict_movie_rating(movie_data, model, vectorizer, scaler, feature_columns):
    """Predict rating for a movie using XGBoost model"""
    features = prepare_movie_features(movie_data, vectorizer, scaler, feature_columns)
    if features is not None:
        prediction = model.predict(features)[0]
        return max(0, min(10, prediction))  # Clamp between 0-10
    return None

# Initialize or load recommendation history
def initialize_history():
    history_file = 'recommendation_history.csv'
    if os.path.exists(history_file) and os.path.getsize(history_file) > 0:
        return pd.read_csv(history_file)
    else:
        history_df = pd.DataFrame(columns=[
            'Date', 'Movie_Title', 'Genre_Filter', 'Rating_Filter', 'Year_Filter', 'Predicted_Rating'
        ])
        history_df.to_csv(history_file, index=False)
        return history_df

# Function to get movie recommendations with XGBoost predictions
def get_recommendations_with_predictions(title, cosine_sim, data, indices, xgb_model, vectorizer, scaler, feature_columns, genre=None, min_rating=0, release_year=None):
    # Get the index of the movie that matches the title
    try:
        idx = indices[title]
    except:
        return pd.DataFrame()
    
    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:21]  # Get more initially for filtering
    
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Get the recommendations
    recommendations = data.iloc[movie_indices].copy()
    
    # Add XGBoost predictions
    predicted_ratings = []
    for _, row in recommendations.iterrows():
        movie_data = {
            'genres': row.get('genres', '[]'),
            'cast': row.get('cast', '[]'),
            'crew': row.get('crew', '[]'),
            'budget': row.get('budget', 0),
            'revenue': row.get('revenue', 0),
            'runtime': row.get('runtime', 90),
            'popularity': row.get('popularity', 1),
            'vote_count': row.get('vote_count', 10)
        }
        
        if xgb_model:
            pred_rating = predict_movie_rating(movie_data, xgb_model, vectorizer, scaler, feature_columns)
            predicted_ratings.append(pred_rating if pred_rating else 0)
        else:
            predicted_ratings.append(0)
    
    recommendations['predicted_rating'] = predicted_ratings
    
    # Apply filters - handle different column names
    filtered = recommendations.copy()
    
    if genre and genre != "All Genres":
        # Try different genre column names
        if 'genres_str' in filtered.columns:
            filtered = filtered[filtered['genres_str'].str.contains(genre, case=False, na=False)]
        elif 'genres' in filtered.columns:
            # Handle list-type genres column
            filtered = filtered[filtered['genres'].apply(lambda x: genre in str(x) if pd.notna(x) else False)]
    
    if min_rating > 0:
        rating_col = 'vote_average' if 'vote_average' in filtered.columns else 'rating'
        if rating_col in filtered.columns:
            filtered = filtered[filtered[rating_col] >= min_rating]
    
    if release_year and release_year != "All Years":
        year_col = None
        if 'release_year' in filtered.columns:
            year_col = 'release_year'
        elif 'year' in filtered.columns:
            year_col = 'year'
        
        if year_col:
            filtered = filtered[filtered[year_col] == int(release_year)]
    
    # Select columns that exist
    result_columns = ['title']
    if 'genres_str' in filtered.columns:
        result_columns.append('genres_str')
    
    rating_col = 'vote_average' if 'vote_average' in filtered.columns else 'rating'
    if rating_col in filtered.columns:
        result_columns.append(rating_col)
    
    year_col = 'release_year' if 'release_year' in filtered.columns else 'year'
    if year_col in filtered.columns:
        result_columns.append(year_col)
    
    result_columns.append('predicted_rating')
    
    return filtered[result_columns].head(10)

# Main function
def main():
    # Load models
    xgb_model, scaler, vectorizer, feature_columns, is_trained = load_xgboost_model()
    tfidf, cosine_sim, indices, data = load_recommendation_model()
    
    if not is_trained:
        st.warning("XGBoost model not properly trained. Rating predictions may not be available.")
    
    if tfidf is None or cosine_sim is None or indices is None or data is None:
        st.error("Failed to load recommendation model components. Please make sure the model files exist.")
        return
    
    # Load history
    history_df = initialize_history()
    
    # App title and description
    st.markdown("<h1 style='text-align: center;'>🎬 Movie Recommendation & Rating System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Find movies similar to your favorites with AI-powered rating predictions!</p>", unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Get Recommendations", "🤖 Predict Rating", "📜 History"])
    
    # Recommendations Tab
    with tab1:
        st.markdown("<h2>Find Your Next Favorite Movie</h2>", unsafe_allow_html=True)
        
        # Movie selection
        movie_list = data['title'].tolist()
        selected_movie = st.selectbox("Select a movie you like:", movie_list)
        
        # Create three columns for filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Extract unique genres - check what column exists
            genre_options = ["All Genres"]
            try:
                if 'genres' in data.columns:
                    all_genres = []
                    for genres in data['genres']:
                        if isinstance(genres, list):
                            all_genres.extend(genres)
                    unique_genres = sorted(list(set(all_genres)))
                    genre_options = ["All Genres"] + unique_genres
                elif 'genres_str' in data.columns:
                    # If genres_str exists, extract from there
                    all_genres = []
                    for genre_str in data['genres_str'].dropna():
                        all_genres.extend([g.strip() for g in str(genre_str).split()])
                    unique_genres = sorted(list(set(all_genres)))
                    genre_options = ["All Genres"] + unique_genres
                else:
                    st.warning("No genre information available in the dataset")
            except Exception as e:
                st.error(f"Error processing genres: {str(e)}")
                st.info(f"Available columns: {list(data.columns)}")
            
            selected_genre = st.selectbox("Filter by Genre:", genre_options)
        
        with col2:
            min_rating = st.slider("Minimum Rating:", 0.0, 10.0, 0.0, 0.5)
        
        with col3:
            # Extract unique years - handle different possible column names
            year_options = ["All Years"]
            try:
                year_column = None
                if 'release_year' in data.columns:
                    year_column = 'release_year'
                elif 'year' in data.columns:
                    year_column = 'year'
                elif 'release_date' in data.columns:
                    # If release_date exists, extract year from it
                    data['release_year'] = pd.to_datetime(data['release_date'], errors='coerce').dt.year
                    year_column = 'release_year'
                
                if year_column:
                    years = sorted([y for y in data[year_column].unique() if pd.notna(y)], reverse=True)
                    year_options = ["All Years"] + [str(int(year)) for year in years]
                else:
                    st.warning("No year information available in the dataset")
            except Exception as e:
                st.error(f"Error processing years: {str(e)}")
                st.info(f"Available columns: {list(data.columns)}")
            
            selected_year = st.selectbox("Release Year:", year_options)
        
        # Get recommendations button
        if st.button("Get Recommendations", type="primary", use_container_width=True):
            with st.spinner("Finding recommendations and predicting ratings..."):
                # Get recommendations with predictions
                year_filter = None if selected_year == "All Years" else selected_year
                genre_filter = None if selected_genre == "All Genres" else selected_genre
                
                recommendations = get_recommendations_with_predictions(
                    selected_movie, 
                    cosine_sim, 
                    data, 
                    indices,
                    xgb_model,
                    vectorizer,
                    scaler,
                    feature_columns,
                    genre=genre_filter, 
                    min_rating=min_rating, 
                    release_year=year_filter
                )
                
                # Save to history
                avg_predicted = recommendations['predicted_rating'].mean() if not recommendations.empty else 0
                new_row = {
                    'Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Movie_Title': selected_movie,
                    'Genre_Filter': selected_genre,
                    'Rating_Filter': min_rating,
                    'Year_Filter': selected_year,
                    'Predicted_Rating': f"{avg_predicted:.1f}"
                }
                
                history_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)
                history_df.to_csv('recommendation_history.csv', index=False)
                
                # Display recommendations
                if not recommendations.empty:
                    st.markdown("<h3>Recommended Movies</h3>", unsafe_allow_html=True)
                    
                    for i, row in recommendations.iterrows():
                        # Determine rating color class - handle different column names
                        rating_col = 'vote_average' if 'vote_average' in row.index else 'rating'
                        year_col = 'release_year' if 'release_year' in row.index else 'year'
                        genres_col = 'genres_str' if 'genres_str' in row.index else 'genres'
                        
                        actual_rating = row.get(rating_col, 0)
                        predicted_rating = row.get('predicted_rating', 0)
                        release_year_val = row.get(year_col, 'Unknown')
                        genres_val = row.get(genres_col, 'Unknown')
                        
                        if actual_rating >= 7.5:
                            rating_class = "rating-high"
                        elif actual_rating >= 6.0:
                            rating_class = "rating-medium"
                        else:
                            rating_class = "rating-low"
                        
                        # Format year display
                        year_display = int(release_year_val) if pd.notna(release_year_val) and str(release_year_val).replace('.','').isdigit() else 'Unknown'
                        
                        # Create a card for each recommendation
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <div class="movie-title">{row['title']}</div>
                            <div class="movie-info">
                                <span>Genres: {genres_val}</span><br>
                                <span>Actual Rating: <span class="{rating_class}">{actual_rating:.1f}/10</span></span><br>
                                <span class="predicted-rating">🤖 AI Predicted: {predicted_rating:.1f}/10</span><br>
                                <span>Release Year: {year_display}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("No recommendations found with the selected filters. Try adjusting your filters.")
    
    # Rating Prediction Tab
    with tab2:
        st.markdown("<h2>Predict Movie Rating with AI</h2>", unsafe_allow_html=True)
        
        if not is_trained:
            st.error("XGBoost model not available for predictions.")
            return
        
        st.markdown('<div class="prediction-tab">', unsafe_allow_html=True)
        
        # Movie input form
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Movie Details")
            pred_budget = st.number_input("Budget ($)", min_value=0, value=50000000, step=1000000)
            pred_revenue = st.number_input("Revenue ($)", min_value=0, value=100000000, step=1000000)
            pred_runtime = st.number_input("Runtime (minutes)", min_value=60, value=120, step=5)
            pred_popularity = st.number_input("Popularity Score", min_value=0.0, value=10.0, step=0.1)
            pred_vote_count = st.number_input("Vote Count", min_value=1, value=1000, step=10)
        
        with col2:
            st.subheader("Cast & Crew")
            pred_genres = st.text_input("Genres (comma-separated)", value="Action, Adventure, Sci-Fi")
            pred_cast = st.text_input("Main Cast (comma-separated)", value="Actor One, Actor Two, Actor Three")
            pred_director = st.text_input("Director", value="Famous Director")
        
        if st.button("Predict Rating", type="primary", use_container_width=True):
            # Prepare movie data for prediction
            movie_data = {
                'budget': pred_budget,
                'revenue': pred_revenue,
                'runtime': pred_runtime,
                'popularity': pred_popularity,
                'vote_count': pred_vote_count,
                'genres': '[' + ', '.join([f'{{"name": "{g.strip()}"}}' for g in pred_genres.split(",")]) + ']',
                'cast': '[' + ', '.join([f'{{"name": "{c.strip()}"}}' for c in pred_cast.split(",")]) + ']',
                'crew': f'[{{"name": "{pred_director.strip()}", "job": "Director"}}]'
            }
            
            # Make prediction
            predicted_rating = predict_movie_rating(movie_data, xgb_model, vectorizer, scaler, feature_columns)
            
            if predicted_rating:
                if predicted_rating >= 7.5:
                    rating_emoji = "🌟"
                    rating_text = "Excellent"
                elif predicted_rating >= 6.0:
                    rating_emoji = "👍"
                    rating_text = "Good"
                else:
                    rating_emoji = "👎"
                    rating_text = "Below Average"
                
                st.markdown(f"""
                <div style="text-align: center; margin: 20px 0;">
                    <h1 style="color: #ff4b4b;">{rating_emoji} {predicted_rating:.1f}/10</h1>
                    <h3>{rating_text} Movie</h3>
                    <p>Based on the provided movie details, our AI model predicts this rating.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("Could not generate prediction. Please check your inputs.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # History Tab
    with tab3:
        st.markdown("<h2>Your Recommendation History</h2>", unsafe_allow_html=True)
        if not history_df.empty:
            st.dataframe(history_df, use_container_width=True)
            
            # Add clear history button
            if st.button("Clear History", type="secondary"):
                empty_df = pd.DataFrame(columns=[
                    'Date', 'Movie_Title', 'Genre_Filter', 'Rating_Filter', 'Year_Filter', 'Predicted_Rating'
                ])
                empty_df.to_csv('recommendation_history.csv', index=False)
                st.success("History cleared!")
                st.rerun()
        else:
            st.info("No recommendation history available yet. Get some recommendations to see them here.")

if __name__ == "__main__":
    main()