import streamlit as st
import pandas as pd
import json
import ast
import plotly.express as px
import plotly.graph_objects as go

# Set page config for a clean layout
st.set_page_config(page_title="TMDb Movies Dashboard", layout="wide")

# Title and introduction
st.title("TMDb Movies and Credits Dashboard")
st.markdown("""
Welcome to the TMDb Movies Dashboard! This app explores the `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` datasets, 
providing insights into movie metadata, cast, crew, and visualizations of key metrics like revenue and ratings.
""")

# Sidebar for file upload and filters
st.sidebar.header("Data and Filters")
uploaded_movies = st.sidebar.file_uploader("Upload tmdb_5000_movies.csv", type="csv")
uploaded_credits = st.sidebar.file_uploader("Upload tmdb_5000_credits.csv", type="csv")

# Function to parse JSON-like columns
def parse_json_column(column):
    try:
        return column.apply(lambda x: [item['name'] for item in ast.literal_eval(x)] if pd.notnull(x) and x != '[]' else [])
    except (ValueError, SyntaxError):
        return column.apply(lambda x: [])

# Function to extract director
def extract_director(crew_str):
    try:
        crew_list = ast.literal_eval(crew_str)
        if crew_list:  # Check if list is not empty
            directors = [item['name'] for item in crew_list if item.get('job') == 'Director']
            return directors[0] if directors else None
        return None
    except (ValueError, SyntaxError):
        return None

# Load sample data if no files are uploaded
if uploaded_movies is None or uploaded_credits is None:
    st.warning("No files uploaded. Using sample data for demonstration.")
    # Sample data for tmdb_5000_movies.csv (added budget)
    movies_data = {
        'id': [19995, 285, 206647, 49026, 24428, 99861, 58, 1865, 120, 671],
        'title': ['Avatar', 'Pirates of the Caribbean: At World\'s End', 'Spectre', 'The Dark Knight Rises', 
                  'The Avengers', 'Avengers: Age of Ultron', 'Pirates of the Caribbean: Dead Man\'s Chest', 
                  'Pirates of the Caribbean: On Stranger Tides', 'Harry Potter and the Half-Blood Prince', 'Spider-Man 3'],
        'budget': [237000000, 300000000, 245000000, 250000000, 220000000, 280000000, 200000000, 250000000, 225000000, 258000000],
        'revenue': [2787965087, 961000000, 880674609, 1084939099, 1519557910, 1405403694, 1065659812, 1045713802, 933959197, 890871626],
        'vote_average': [7.2, 6.9, 6.3, 7.6, 7.4, 7.3, 7.0, 6.4, 7.4, 5.9],
        'vote_count': [11800, 4500, 4466, 9106, 11776, 6767, 4238, 4948, 5293, 3576],
        'genres': [
            '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]',
            '[{"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 28, "name": "Action"}]',
            '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 80, "name": "Crime"}]',
            '[{"id": 28, "name": "Action"}, {"id": 80, "name": "Crime"}, {"id": 18, "name": "Drama"}, {"id": 53, "name": "Thriller"}]',
            '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 878, "name": "Science Fiction"}]',
            '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 878, "name": "Science Fiction"}]',
            '[{"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 28, "name": "Action"}]',
            '[{"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 28, "name": "Action"}]',
            '[{"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 10751, "name": "Family"}]',
            '[{"id": 14, "name": "Fantasy"}, {"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]'
        ]
    }
    movies_df = pd.DataFrame(movies_data)
    
    # Sample data for tmdb_5000_credits.csv
    credits_data = {
        'movie_id': [19995, 285, 206647, 49026, 24428, 99861, 58, 1865, 120, 671],
        'title': ['Avatar', 'Pirates of the Caribbean: At World\'s End', 'Spectre', 'The Dark Knight Rises', 
                  'The Avengers', 'Avengers: Age of Ultron', 'Pirates of the Caribbean: Dead Man\'s Chest', 
                  'Pirates of the Caribbean: On Stranger Tides', 'Harry Potter and the Half-Blood Prince', 'Spider-Man 3'],
        'crew': [
            '[{"job": "Director", "name": "James Cameron"}]',
            '[{"job": "Director", "name": "Gore Verbinski"}]',
            '[{"job": "Director", "name": "Sam Mendes"}]',
            '[{"job": "Director", "name": "Christopher Nolan"}]',
            '[{"job": "Director", "name": "Joss Whedon"}]',
            '[{"job": "Director", "name": "Joss Whedon"}]',
            '[{"job": "Director", "name": "Gore Verbinski"}]',
            '[{"job": "Director", "name": "Rob Marshall"}]',
            '[{"job": "Director", "name": "David Yates"}]',
            '[{"job": "Director", "name": "Sam Raimi"}]'
        ]
    }
    credits_df = pd.DataFrame(credits_data)
else:
    movies_df = pd.read_csv(uploaded_movies)
    credits_df = pd.read_csv(uploaded_credits)

# Merge datasets
merged_df = movies_df.merge(credits_df, left_on='id', right_on='movie_id', how='inner', suffixes=('_movies', '_credits'))

# Parse genres and crew
merged_df['genres_list'] = parse_json_column(merged_df['genres'])
merged_df['director'] = merged_df['crew'].apply(extract_director)

# Display movies with missing directors
missing_directors = merged_df[merged_df['director'].isnull()][['title_movies', 'crew']]
if not missing_directors.empty:
    st.warning(f"Found {len(missing_directors)} movies with missing or invalid director data. Check the 'crew' column for these movies:")
    st.dataframe(missing_directors)

# Tabs for different sections
tab1, tab2, tab3 = st.tabs(["Dataset Overview", "Visualizations", "Interactive Exploration"])

with tab1:
    st.header("Dataset Overview")
    st.markdown("### TMDb Movies Dataset (`tmdb_5000_movies.csv`)")
    st.markdown("""
    - **Rows**: 4,803
    - **Columns**: 20
    - **Key Columns**:
        - `budget`: Budget in USD (int)
        - `genres`: JSON list of genres (e.g., Action, Adventure)
        - `id`: Unique movie ID (int)
        - `revenue`: Revenue in USD (int)
        - `vote_average`: Average rating (0-10, float)
        - `vote_count`: Number of votes (int)
        - Others: `title`, `popularity`, `release_date`, etc.
    - **Insights**: Includes blockbusters (e.g., *Avatar*: $2.79B revenue) and indie films (e.g., *Clerks*: $27K budget).
    """)
    # Dynamically select available columns
    movie_columns = ['id', 'title', 'budget', 'revenue', 'vote_average', 'vote_count', 'genres']
    available_columns = [col for col in movie_columns if col in movies_df.columns]
    st.write("Sample Data (Movies):")
    st.dataframe(movies_df[available_columns].head())

    st.markdown("### TMDb Credits Dataset (`tmdb_5000_credits.csv`)")
    st.markdown("""
    - **Rows**: 4,803
    - **Columns**: 4
    - **Key Columns**:
        - `movie_id`: Matches `id` in movies dataset
        - `title`: Movie title
        - `cast`: JSON list of actors (name, character, etc.)
        - `crew`: JSON list of crew (director, writer, etc.)
    - **Insights**: Enables analysis of directors (e.g., James Cameron for *Avatar*) and actors.
    """)
    st.write("Sample Data (Credits):")
    st.dataframe(credits_df[['movie_id', 'title', 'crew']].head())

with tab2:
    st.header("Visualizations")
    
    # Top 10 Movies by Revenue
    st.subheader("Top 10 Movies by Revenue")
    top_movies = movies_df.nlargest(10, 'revenue')[['title', 'revenue']].sort_values(by='revenue', ascending=False)
    top_movies['revenue_billions'] = top_movies['revenue'] / 1e9
    fig1 = px.bar(top_movies, x='title', y='revenue_billions', title="Top 10 Movies by Revenue",
                  labels={'title': 'Movie Title', 'revenue_billions': 'Revenue (Billions USD)'},
                  color='title', color_discrete_sequence=px.colors.qualitative.Plotly)
    fig1.update_layout(showlegend=False, xaxis_tickangle=45)
    st.plotly_chart(fig1, use_container_width=True)

    # Top Genres by Average Rating
    st.subheader("Top Genres by Average Rating")
    genre_ratings = []
    for index, row in merged_df.iterrows():
        for genre in row['genres_list']:
            if row['vote_count'] >= 50:  # Filter for reliable ratings
                genre_ratings.append({'genre': genre, 'vote_average': row['vote_average']})
    genre_df = pd.DataFrame(genre_ratings)
    genre_avg = genre_df.groupby('genre')['vote_average'].agg(['mean', 'count']).reset_index()
    genre_avg = genre_avg[genre_avg['count'] >= 3].sort_values(by='mean', ascending=False)
    fig2 = px.bar(genre_avg, x='genre', y='mean', title="Top Genres by Average Rating",
                  labels={'genre': 'Genre', 'mean': 'Average Rating (0-10)'},
                  color='genre', color_discrete_sequence=px.colors.qualitative.Plotly)
    fig2.update_layout(showlegend=False, xaxis_tickangle=45, yaxis_range=[0, 10])
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.header("Interactive Exploration")
    st.markdown("Filter movies by genre and minimum vote count.")
    
    # Genre filter
    all_genres = sorted(set([genre for genres in merged_df['genres_list'] for genre in genres]))
    selected_genre = st.selectbox("Select Genre", ["All"] + all_genres)
    min_votes = st.slider("Minimum Vote Count", 0, 1000, 50, 10)
    
    # Filter data
    filtered_df = merged_df[merged_df['vote_count'] >= min_votes]
    if selected_genre != "All":
        filtered_df = filtered_df[filtered_df['genres_list'].apply(lambda x: selected_genre in x)]
    
    st.write(f"Filtered Movies ({len(filtered_df)} results):")
    st.dataframe(filtered_df[['title_movies', 'director', 'vote_average', 'revenue', 'genres_list']])
    
    # Director frequency
    st.subheader("Top Directors by Movie Count")
    director_counts = filtered_df['director'].value_counts().reset_index().head(10)
    director_counts.columns = ['director', 'movie_count']
    fig3 = px.bar(director_counts, x='director', y='movie_count', title="Top Directors by Movie Count",
                  labels={'director': 'Director', 'movie_count': 'Number of Movies'},
                  color='director', color_discrete_sequence=px.colors.qualitative.Plotly)
    fig3.update_layout(showlegend=False, xaxis_tickangle=45)
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")
st.markdown("Built with Streamlit. Upload the full datasets for comprehensive analysis!")