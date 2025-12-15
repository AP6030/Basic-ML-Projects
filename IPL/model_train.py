import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse

# Load the dataset
data = pd.read_csv('c:\\Users\\ASUS\\Intern-Pe-Updated\\IPL\\ipl_prediction.csv')
print(f"Dataset successfully imported with shape: {data.shape}")

# Data preprocessing
# Label encoding for categorical variables
le = LabelEncoder()
for col in ['batting_team', 'bowling_team']:
    data[col] = le.fit_transform(data[col])

# One-hot encoding for categorical variables
columnTransformer = ColumnTransformer([('encoder',
                                      OneHotEncoder(),
                                      [0, 1])],
                                    remainder='passthrough')

data = np.array(columnTransformer.fit_transform(data))

# Define column names after transformation
cols = ['batting_team_Chennai Super Kings', 'batting_team_Delhi Daredevils', 'batting_team_Kings XI Punjab',
        'batting_team_Kolkata Knight Riders', 'batting_team_Mumbai Indians', 'batting_team_Rajasthan Royals',
        'batting_team_Royal Challengers Bangalore', 'batting_team_Sunrisers Hyderabad',
        'bowling_team_Chennai Super Kings', 'bowling_team_Delhi Daredevils', 'bowling_team_Kings XI Punjab',
        'bowling_team_Kolkata Knight Riders', 'bowling_team_Mumbai Indians', 'bowling_team_Rajasthan Royals',
        'bowling_team_Royal Challengers Bangalore', 'bowling_team_Sunrisers Hyderabad', 'runs', 'wickets', 'overs',
        'runs_last_5', 'wickets_last_5', 'total']

df = pd.DataFrame(data, columns=cols)

# Feature selection and target variable
features = df.drop(['total'], axis=1)
labels = df['total']

# Train-test split
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.20, shuffle=True)
print(f"Training Set : {train_features.shape}\nTesting Set : {test_features.shape}")

# Train the Random Forest model (best performing model)
forest = RandomForestRegressor()
forest.fit(train_features, train_labels)

# Evaluate the model
train_score_forest = forest.score(train_features, train_labels) * 100
test_score_forest = forest.score(test_features, test_labels) * 100
print(f'Train Score : {train_score_forest:.2f}%\nTest Score : {test_score_forest:.2f}%')

# Calculate error metrics
print("---- Random Forest Regression - Model Evaluation ----")
print(f"Mean Absolute Error (MAE): {mae(test_labels, forest.predict(test_features))}")
print(f"Mean Squared Error (MSE): {mse(test_labels, forest.predict(test_features))}")
print(f"Root Mean Squared Error (RMSE): {np.sqrt(mse(test_labels, forest.predict(test_features)))}")

# Function to predict score
def predict_score(batting_team, bowling_team, runs, wickets, overs, runs_last_5, wickets_last_5):
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
    
    # Add match stats
    prediction_array = prediction_array + [runs, wickets, overs, runs_last_5, wickets_last_5]
    
    # Make prediction
    prediction_array = np.array(prediction_array).reshape(1, -1)
    return forest.predict(prediction_array)[0]

# Save the model
with open('ipl_model.pkl', 'wb') as f:
    pickle.dump(forest, f)

print("Model saved as 'ipl_model.pkl'")

# Example prediction
teams = ['Chennai Super Kings', 'Delhi Daredevils', 'Kings XI Punjab', 'Kolkata Knight Riders', 
         'Mumbai Indians', 'Rajasthan Royals', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad']

print("\nExample prediction:")
print(f"Predicted score: {predict_score('Mumbai Indians', 'Royal Challengers Bangalore', 100, 2, 12.0, 40, 1):.2f}")