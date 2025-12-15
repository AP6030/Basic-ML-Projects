import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime

# Load the Random Forest model
def load_model():
    try:
        # If you have a saved model, load it
        # with open('ipl_model.pkl', 'rb') as f:
        #     model = pickle.load(f)
        # return model
        
        # Since we don't have a saved model yet, we'll train it here
        # This is a placeholder - you should save your model from the notebook
        # and load it here instead of training it every time
        # Change to:
        data = pd.read_csv('IPL/ipl_prediction.csv')
        # Implement model training code here or load from pickle file
        # For now, we'll return None and handle it later
        return None
    except Exception as e:
        messagebox.showerror("Error", f"Error loading model: {str(e)}")
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

# Function to make prediction and update history
def make_prediction():
    try:
        # Get input values
        batting_team = batting_team_var.get()
        bowling_team = bowling_team_var.get()
        runs = float(entry_runs.get())
        wickets = int(entry_wickets.get())
        overs = float(entry_overs.get())
        runs_last_5 = float(entry_runs_last_5.get())
        wickets_last_5 = int(entry_wickets_last_5.get())
        
        # Validate inputs
        if batting_team == bowling_team:
            messagebox.showerror("Error", "Batting team and bowling team cannot be the same")
            return
            
        if overs > 20 or overs < 0:
            messagebox.showerror("Error", "Overs must be between 0 and 20")
            return
            
        if wickets > 10 or wickets < 0:
            messagebox.showerror("Error", "Wickets must be between 0 and 10")
            return
            
        if wickets_last_5 > 5 or wickets_last_5 < 0:
            messagebox.showerror("Error", "Wickets in last 5 overs must be between 0 and 5")
            return
        
        # Make prediction
        # For now, we'll use a placeholder prediction since we don't have the model loaded
        # In a real implementation, you would use your trained model
        # predicted_score = predict_score(batting_team, bowling_team, runs, wickets, overs, runs_last_5, wickets_last_5, model)
        
        # Placeholder prediction (you should replace this with actual model prediction)
        predicted_score = int(runs + (20 - overs) * (runs / overs) * 0.8)
        
        # Display result
        result_label.config(text=f"Predicted Final Score: {predicted_score} runs")
        
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
        
        global history_df
        history_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)
        history_df.to_csv('ipl_prediction_history.csv', index=False)
        
        # Update history table
        update_history_table()
        
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Function to update history table
def update_history_table():
    # Clear existing data
    for item in history_tree.get_children():
        history_tree.delete(item)
    
    # Add data to treeview
    for i, row in history_df.iterrows():
        values = [
            row['Date'],
            row['Batting_Team'],
            row['Bowling_Team'],
            row['Runs'],
            row['Wickets'],
            row['Overs'],
            row['Runs_Last_5'],
            row['Wickets_Last_5'],
            row['Predicted_Score']
        ]
        history_tree.insert("", "end", values=values)

# Main application
if __name__ == "__main__":
    # Load model and initialize history
    model = load_model()
    history_df = initialize_history()
    
    # Create main window
    root = tk.Tk()
    root.title("IPL Score Predictor")
    root.geometry("1000x700")
    
    # Create notebook (tabs)
    nb = ttk.Notebook(root)
    nb.pack(fill='both', expand=True, padx=10, pady=10)
    
    # Prediction Frame
    prediction_frame = ttk.Frame(nb)
    nb.add(prediction_frame, text="Make Prediction")
    
    # History Frame
    history_frame = ttk.Frame(nb)
    nb.add(history_frame, text="Prediction History")
    
    # Input frame
    input_frame = ttk.LabelFrame(prediction_frame, text="Match Information")
    input_frame.pack(fill="x", padx=10, pady=10)
    
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
    
    # Batting team
    tk.Label(input_frame, text="Batting Team").grid(row=0, column=0, padx=10, pady=5, sticky='e')
    batting_team_var = tk.StringVar()
    batting_team_dropdown = ttk.Combobox(input_frame, textvariable=batting_team_var, values=teams, width=30)
    batting_team_dropdown.grid(row=0, column=1, padx=10, pady=5)
    batting_team_dropdown.current(0)
    
    # Bowling team
    tk.Label(input_frame, text="Bowling Team").grid(row=1, column=0, padx=10, pady=5, sticky='e')
    bowling_team_var = tk.StringVar()
    bowling_team_dropdown = ttk.Combobox(input_frame, textvariable=bowling_team_var, values=teams, width=30)
    bowling_team_dropdown.grid(row=1, column=1, padx=10, pady=5)
    bowling_team_dropdown.current(1)
    
    # Current match stats
    tk.Label(input_frame, text="Current Runs").grid(row=2, column=0, padx=10, pady=5, sticky='e')
    entry_runs = tk.Entry(input_frame, width=30)
    entry_runs.grid(row=2, column=1, padx=10, pady=5)
    entry_runs.insert(0, "0")
    
    tk.Label(input_frame, text="Current Wickets").grid(row=3, column=0, padx=10, pady=5, sticky='e')
    entry_wickets = tk.Entry(input_frame, width=30)
    entry_wickets.grid(row=3, column=1, padx=10, pady=5)
    entry_wickets.insert(0, "0")
    
    tk.Label(input_frame, text="Current Overs").grid(row=4, column=0, padx=10, pady=5, sticky='e')
    entry_overs = tk.Entry(input_frame, width=30)
    entry_overs.grid(row=4, column=1, padx=10, pady=5)
    entry_overs.insert(0, "0")
    
    tk.Label(input_frame, text="Runs in Last 5 Overs").grid(row=5, column=0, padx=10, pady=5, sticky='e')
    entry_runs_last_5 = tk.Entry(input_frame, width=30)
    entry_runs_last_5.grid(row=5, column=1, padx=10, pady=5)
    entry_runs_last_5.insert(0, "0")
    
    tk.Label(input_frame, text="Wickets in Last 5 Overs").grid(row=6, column=0, padx=10, pady=5, sticky='e')
    entry_wickets_last_5 = tk.Entry(input_frame, width=30)
    entry_wickets_last_5.grid(row=6, column=1, padx=10, pady=5)
    entry_wickets_last_5.insert(0, "0")
    
    # Predict button
    predict_button = tk.Button(input_frame, text="Predict Score", command=make_prediction, 
                             bg='green', fg='white', font=('Arial', 12, 'bold'))
    predict_button.grid(row=7, column=0, columnspan=2, pady=20)
    
    # Result label
    result_label = tk.Label(prediction_frame, text="Enter match details and click 'Predict Score'", 
                           font=("Arial", 14, "bold"), fg="blue")
    result_label.pack(pady=20)
    
    # History table
    history_tree = ttk.Treeview(history_frame)
    history_tree["columns"] = (
        "Date", "Batting_Team", "Bowling_Team", "Runs", "Wickets", 
        "Overs", "Runs_Last_5", "Wickets_Last_5", "Predicted_Score"
    )
    
    # Configure columns
    history_tree.column("#0", width=0, stretch=tk.NO)
    for col in history_tree["columns"]:
        history_tree.column(col, anchor=tk.CENTER, width=100)
        history_tree.heading(col, text=col)
    
    # Add scrollbar
    scrollbar = ttk.Scrollbar(history_frame, orient="vertical", command=history_tree.yview)
    history_tree.configure(yscrollcommand=scrollbar.set)
    
    # Pack history components
    history_tree.pack(side="left", fill="both", expand=True, padx=10, pady=10)
    scrollbar.pack(side="right", fill="y", pady=10)
    
    # Initialize history table
    update_history_table()
    
    # Start the application
    root.mainloop()