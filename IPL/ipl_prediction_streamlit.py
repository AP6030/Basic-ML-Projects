import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime

# Custom CSS for blue theme
# Custom CSS for blue theme
def apply_blue_theme():
    st.markdown("""
    <style>
    /* Main background and text colors */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #4da6ff !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-12oz5g7 {
        background-color: #1a2634;
    }
    
    /* Input fields */
    .stTextInput > div > div > input, .stNumberInput > div > div > input {
        background-color: #1e2a3a;
        color: white;
        border-color: #4da6ff;
    }
    
    /* Selectbox */
    .stSelectbox > div > div > div {
        background-color: #1e2a3a;
        color: white;
    }
    
    /* Button styling with gradient */
    .stButton > button {
        background: linear-gradient(135deg, #0052a3, #0066cc, #4da6ff);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #4da6ff, #0066cc, #0052a3);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.2);
        transform: translateY(-2px);
    }
    
    /* Success message */
    .element-container div[data-testid="stAlert"] {
        background-color: #1e3a8a;
        color: white;
        border: 1px solid #4da6ff;
        border-radius: 5px;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1a2634;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: white;
        border-radius: 10px 10px 0 0;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0052a3, #0066cc);
        color: white;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border: 1px solid #4da6ff;
    }
    
    .dataframe {
        background-color: #1e2a3a;
        color: white;
    }
    
    .dataframe th {
        background-color: #0066cc;
        color: white;
    }
    
    /* Info box */
    div[data-testid="stInfoBox"] {
        background-color: #1e3a8a;
        color: white;
        border: 1px solid #4da6ff;
    }
    
    /* Gradient for prediction result */
    .prediction-result {
        background: linear-gradient(135deg, #0a2e5c, #1e3a8a, #0a2e5c);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #4da6ff;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        animation: gradient-shift 5s ease infinite;
    }
    
    /* Gradient for dataframe container */
    .dataframe-container {
        background: linear-gradient(135deg, #0a2e5c, #1a2634, #0a2e5c);
        border: 2px solid #4da6ff;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        animation: gradient-shift 5s ease infinite;
    }
    
    /* No history message styling */
    .no-history {
        background: linear-gradient(135deg, #0a2e5c, #1e3a8a, #0a2e5c);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #4da6ff;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        animation: gradient-shift 5s ease infinite;
    }
    
    /* Animation for gradient shift */
    @keyframes gradient-shift {
        0% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 50%;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Set page config
st.set_page_config(page_title="IPL Score Predictor", layout="wide")

# Apply the blue theme
apply_blue_theme()

# Load the Random Forest model
def load_model():
    try:
        # Load the saved model
        with open('IPL/forest_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Initialize or load prediction history
def initialize_history():
    history_file = 'ipl_prediction_history.csv'
    if os.path.exists(history_file) and os.path.getsize(history_file) > 0:
        return pd.read_csv(history_file)
    else:
        history_df = pd.DataFrame(columns=[
            'Date', 'Batting_Team', 'Bowling_Team', 'Runs', 'Wickets', 
            'Overs', 'Runs_Last_5', 'Wickets_Last_5', 'Predicted_Score'
        ])
        history_df.to_csv(history_file, index=False)
        return history_df

# Prediction function from your notebook
def predict_score(batting_team, bowling_team, runs, wickets, overs, runs_last_5, wickets_last_5, model):
    prediction_array = []
    # Batting Team
    if batting_team == 'Chennai Super Kings':
        prediction_array = prediction_array + [1,0,0,0,0,0,0,0]
    elif batting_team == 'Delhi Daredevils':
        prediction_array = prediction_array + [0,1,0,0,0,0,0,0]
    elif batting_team == 'Kings XI Punjab':
        prediction_array = prediction_array + [0,0,1,0,0,0,0,0]
    elif batting_team == 'Kolkata Knight Riders':
        prediction_array = prediction_array + [0,0,0,1,0,0,0,0]
    elif batting_team == 'Mumbai Indians':
        prediction_array = prediction_array + [0,0,0,0,1,0,0,0]
    elif batting_team == 'Rajasthan Royals':
        prediction_array = prediction_array + [0,0,0,0,0,1,0,0]
    elif batting_team == 'Royal Challengers Bangalore':
        prediction_array = prediction_array + [0,0,0,0,0,0,1,0]
    elif batting_team == 'Sunrisers Hyderabad':
        prediction_array = prediction_array + [0,0,0,0,0,0,0,1]
    # Bowling Team
    if bowling_team == 'Chennai Super Kings':
        prediction_array = prediction_array + [1,0,0,0,0,0,0,0]
    elif bowling_team == 'Delhi Daredevils':
        prediction_array = prediction_array + [0,1,0,0,0,0,0,0]
    elif bowling_team == 'Kings XI Punjab':
        prediction_array = prediction_array + [0,0,1,0,0,0,0,0]
    elif bowling_team == 'Kolkata Knight Riders':
        prediction_array = prediction_array + [0,0,0,1,0,0,0,0]
    elif bowling_team == 'Mumbai Indians':
        prediction_array = prediction_array + [0,0,0,0,1,0,0,0]
    elif bowling_team == 'Rajasthan Royals':
        prediction_array = prediction_array + [0,0,0,0,0,1,0,0]
    elif bowling_team == 'Royal Challengers Bangalore':
        prediction_array = prediction_array + [0,0,0,0,0,0,1,0]
    elif bowling_team == 'Sunrisers Hyderabad':
        prediction_array = prediction_array + [0,0,0,0,0,0,0,1]
    prediction_array = prediction_array + [runs, wickets, overs, runs_last_5, wickets_last_5]
    prediction_array = np.array([prediction_array])
    pred = model.predict(prediction_array)
    return int(round(pred[0]))

# Main function
def main():
    # Load model and initialize history
    model = load_model()
    history_df = initialize_history()
    
    # App title and description with blue styling
    st.markdown("<h1 style='text-align: center; color: #4da6ff;'>IPL Score Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #a6c9ff;'>Predict the final score of an IPL match based on current match statistics.</p>", unsafe_allow_html=True)
    
    # Add a divider
    st.markdown("<hr style='height:2px;border:none;color:#4da6ff;background-color:#4da6ff;' />", unsafe_allow_html=True)
    
    # Create tabs with custom styling
    tab1, tab2 = st.tabs([" Prediction", "📜 Prediction History"])
    
    # Prediction Tab
    with tab1:
        st.markdown("<h2 style='color: #a6c9ff;'>Match Information</h2>", unsafe_allow_html=True)
        
        # Team selection
        teams = [
            'Chennai Super Kings',
            'Delhi Daredevils',
            'Kings XI Punjab',
            'Kolkata Knight Riders',
            'Mumbai Indians',
            'Rajasthan Royals',
            'Royal Challengers Bangalore',
            'Sunrisers Hyderabad'
        ]
        
        # Create two columns for team selection
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<p style='color: #a6c9ff; font-weight: bold;'>Batting Team</p>", unsafe_allow_html=True)
            batting_team = st.selectbox("Batting Team", teams, index=0, label_visibility="collapsed")
        
        with col2:
            st.markdown("<p style='color: #a6c9ff; font-weight: bold;'>Bowling Team</p>", unsafe_allow_html=True)
            bowling_team = st.selectbox("Bowling Team", teams, index=1, label_visibility="collapsed")
        
        # Add a divider
        st.markdown("<hr style='height:1px;border:none;color:#4da6ff;background-color:#4da6ff;' />", unsafe_allow_html=True)
        
        # Create two columns for match stats
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<p style='color: #a6c9ff; font-weight: bold;'>Current Runs</p>", unsafe_allow_html=True)
            runs = st.number_input("Current Runs", min_value=0, value=0, label_visibility="collapsed")
            
            st.markdown("<p style='color: #a6c9ff; font-weight: bold;'>Current Wickets</p>", unsafe_allow_html=True)
            wickets = st.number_input("Current Wickets", min_value=0, max_value=10, value=0, label_visibility="collapsed")
            
            st.markdown("<p style='color: #a6c9ff; font-weight: bold;'>Current Overs</p>", unsafe_allow_html=True)
            overs = st.number_input("Current Overs", min_value=0.0, max_value=20.0, value=0.0, step=0.1, label_visibility="collapsed")
        
        with col2:
            st.markdown("<p style='color: #a6c9ff; font-weight: bold;'>Runs in Last 5 Overs</p>", unsafe_allow_html=True)
            runs_last_5 = st.number_input("Runs in Last 5 Overs", min_value=0, value=0, label_visibility="collapsed")
            
            st.markdown("<p style='color: #a6c9ff; font-weight: bold;'>Wickets in Last 5 Overs</p>", unsafe_allow_html=True)
            wickets_last_5 = st.number_input("Wickets in Last 5 Overs", min_value=0, max_value=5, value=0, label_visibility="collapsed")
        
        # Add some space
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Center the predict button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Predict button
            predict_clicked = st.button("Predict Score", type="primary", use_container_width=True)
        
        if predict_clicked:
            # Validate inputs
            if batting_team == bowling_team:
                st.error("Batting team and bowling team cannot be the same")
            else:
                try:
                    # Make prediction
                    if model is not None:
                        predicted_score = predict_score(batting_team, bowling_team, runs, wickets, overs, runs_last_5, wickets_last_5, model)
                    else:
                        # Placeholder prediction if model isn't loaded
                        predicted_score = int(runs + (20 - overs) * (runs / max(overs, 1)) * 0.8)
                    
                    # Display result with custom styling
                    st.markdown(f"<div class='prediction-result'>"
                                f"<h2 style='color: white;'>Predicted Final Score</h2>"
                                f"<h1 style='color: #4da6ff; font-size: 48px;'>{predicted_score}</h1>"
                                f"<p style='color: white;'>runs</p>"
                                f"</div>", unsafe_allow_html=True)
                    
                    # Save prediction to history
                    new_row = {
                        'Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'Batting_Team': batting_team,
                        'Bowling_Team': bowling_team,
                        'Runs': runs,
                        'Wickets': wickets,
                        'Overs': overs,
                        'Runs_Last_5': runs_last_5,
                        'Wickets_Last_5': wickets_last_5,
                        'Predicted_Score': predicted_score
                    }
                    
                    history_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)
                    history_df.to_csv('ipl_prediction_history.csv', index=False)
                    
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
    
    # History Tab
    with tab2:
        st.markdown("<h2 style='color: #a6c9ff;'>Prediction History</h2>", unsafe_allow_html=True)
        if not history_df.empty:
            st.markdown("<div class='dataframe-container'>", unsafe_allow_html=True)
            st.dataframe(history_df, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='no-history'>"
                        "<p style='color: white;'>No prediction history available yet. Make a prediction to see it here.</p>"
                        "</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()